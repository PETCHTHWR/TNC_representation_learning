import pandas as pd
import numpy as np
from tqdm import tqdm
import connectorx as cx
from atfm.preprocess import preprocess
from atfm.utils import check_eligible, calculate_unit_vectors, is_smooth, calculate_bearing, check_eligible_cupy, is_flight_path_smooth
from sqlalchemy import create_engine

# Query parameters
db_url = 'mysql://lics:aelics070@143.248.69.46:13306/atfm_new'
id_tab = 'flight'
ADSB_tab = 'trajectory'
too_short = 500
too_long = 3000

# Filtering parameters
icn_lat, icn_lon, icn_alt = 37.463333, 126.440002, 8.0
min_alt_change = 2000 / 3.281 # meters
FAF = 1600 / 3.281 # meters
app_sec_rad = 1 # nautical miles
max_cutoff_range = 150 # kilometers
allowed_cutoff_error = 25 # kilometers
max_angle_change = 60 # degrees
max_fltpath_change = 20 # meters
# Preprocessing parameters
target_length = 2000 + 1
data_size = 100


ids_arr = cx.read_sql(db_url,
                      "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=1 AND id <= %d AND id >= %d" % (id_tab, too_short, too_long, id_max, id_min), return_type="arrow")
ids_arr = ids_arr.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates()
ADSB_arr = cx.read_sql(db_url,
                       f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_arr.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_arr = ADSB_arr.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_arr['time'] = pd.to_datetime(ADSB_arr['time'], unit='s')
ADSB_arr = ADSB_arr.groupby('flight_id')
print('Total Arrival :', ids_arr.shape[0])

ids_dep = cx.read_sql(db_url,
                      "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=0 AND id <= %d AND id >= %d" % (id_tab, too_short, too_long, id_max, id_min), return_type="arrow")
ids_dep = ids_dep.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates()
ADSB_dep = cx.read_sql(db_url,
                       f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_dep.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_dep = ADSB_dep.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_dep['time'] = pd.to_datetime(ADSB_dep['time'], unit='s')
ADSB_dep = ADSB_dep.groupby('flight_id')
print('Total Departure :', ids_dep.shape[0])


sql_engine = create_engine('mysql+pymysql://lics:aelics070@143.248.69.46:13306/atfm_new', pool_recycle=3600)
db_connection = sql_engine.connect()

arr_dep = []
for ADSB in [ADSB_arr, ADSB_dep]:
    num_reject = 0
    num_dump = 0
    df_ls = []
    arrival = True if ADSB is ADSB_arr else False
    total_times = []
    for id, flt_df in tqdm(ADSB, total=len(ADSB), desc="Processing groups", ncols=100):

        #flt_df = ADSB.loc[ADSB['flight_id'] == id] # flight dataframe
        flt_df = flt_df.set_index('time') # set time as index

        # check if the flight is eligible
        if not check_eligible_cupy(flt_df, min_alt_change, FAF, icn_lat, icn_lon, app_sec_rad, alt_col='baroaltitude', max_range=max_cutoff_range):
            num_reject += 1
            continue

        # preprocess the flight
        try:
            flt_df = flt_df.dropna() # drop nan values
            flt_df = preprocess(flt_df, ref_lat=icn_lat, ref_lon=icn_lon, ref_alt=icn_alt, periods=target_length, max_range=max_cutoff_range,
                                alt_column='baroaltitude', freq='1s', zscore_threshold=3, window_size=3, order=2) # preprocess

        except Exception: # if preprocessing fails
            num_reject += 1
            continue

        # Print total time in sec
        max_dist = np.max(np.linalg.norm(flt_df[['x', 'y']].values, axis=1))
        if max_dist < (max_cutoff_range - allowed_cutoff_error) * 1000 or max_dist > max_cutoff_range * 1000 or np.min(flt_df['z'].values) > FAF:
            num_reject += 1
            continue

        # Find total time in flight
        total_times.append(flt_df.index[-1] - flt_df.index[0])

        # Under-sampling
        north_bearing = calculate_bearing(flt_df.iloc[-1]['x'], flt_df.iloc[-1]['y'])
        if arrival:
            undersampling_rate = 0.8 # Perform undersampling on southbound and south-eastbound flights
            if north_bearing > 120 and north_bearing < 210:
                if np.random.choice([True, False], p=[undersampling_rate, 1 - undersampling_rate]):
                    num_dump += 1
                    continue
        else:
            undersampling_rate = 0.9 # Perform undersampling on southbound flights
            if north_bearing > 150 and north_bearing < 210:
                if np.random.choice([True, False], p=[undersampling_rate, 1 - undersampling_rate]):
                    num_dump += 1
                    continue
        if arrival:
            undersampling_rate = 0.9 # Perform undersampling on other flights except for eastbound flights
            if north_bearing > 120 and north_bearing < 360:
                if np.random.choice([True, False], p=[undersampling_rate, 1 - undersampling_rate]):
                    num_dump += 1
                    continue
        else:
            undersampling_rate = 0.7 # Perform undersampling on other flights except for westbound flights
            if north_bearing > 0 and north_bearing < 210:
                if np.random.choice([True, False], p=[undersampling_rate, 1 - undersampling_rate]):
                    num_dump += 1
                    continue

        # Calculate unit vectors
        unit_df = calculate_unit_vectors(flt_df)
        if not is_smooth(unit_df, max_angle_change) or not is_flight_path_smooth(unit_df, max_fltpath_change):
            num_reject += 1
            continue

        joined_df = pd.concat([flt_df, unit_df], axis=1) # join the dataframes
        joined_df[unit_df.columns] = joined_df[unit_df.columns].shift(-1) # shift the unit vectors by 1
        df_ls.append(joined_df) # append the dataframe
        db_connection.execute("UPDATE flight SET is_usable=1 WHERE id=%d" % (id))

    print('Rejected %d flights' % num_reject) # print the number of rejected flights
    print('Dumped %d flights' % num_dump) # print the number of dumped flights
    print('Total accepted flights :', len(df_ls)) # print the number of accepted flights
    print('Max_total_time :', np.max(total_times))
    arr_dep.append(df_ls) # arr_dep[0] is arrival, arr_dep[1] is departure


ids_arr = cx.read_sql(db_url,
                      "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=1 AND id <= %d AND id >= %d" % (id_tab, too_short, too_long, id_max, id_min), return_type="arrow")
ids_arr = ids_arr.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates()
ADSB_arr = cx.read_sql(db_url,
                       f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_arr.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_arr = ADSB_arr.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_arr['time'] = pd.to_datetime(ADSB_arr['time'], unit='s')
print('Total Arrival :', ids_arr.shape[0])

ids_dep = cx.read_sql(db_url,
                      "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=0 AND id <= %d AND id >= %d" % (id_tab, too_short, too_long, id_max, id_min), return_type="arrow")
ids_dep = ids_dep.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates()
ADSB_dep = cx.read_sql(db_url,
                       f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_dep.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_dep = ADSB_dep.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_dep['time'] = pd.to_datetime(ADSB_dep['time'], unit='s')
print('Total Departure :', ids_dep.shape[0])