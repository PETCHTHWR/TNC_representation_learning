import pandas as pd
import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import connectorx as cx
from atfm.preprocess import preprocess
from atfm.utils import check_eligible, calculate_unit_vectors, is_smooth, calculate_bearing, check_eligible_cupy, is_flight_path_smooth
from sklearn.preprocessing import MinMaxScaler

def normalize_fourth(train_data, test_data):
    """Normalize only the fourth feature in the train and test data arrays using MinMaxScaler"""

    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Reshape the fourth feature of train data to 2D (samples, length)
    train_fourth_feature = np.reshape(train_data[:, 3, :], (train_data.shape[0], -1))

    # Fit the scaler on the train fourth feature and normalize it
    scaler.fit(train_fourth_feature)
    train_fourth_feature_n = scaler.transform(train_fourth_feature)

    # Normalize the fourth feature of the test data using the fitted scaler
    test_fourth_feature = np.reshape(test_data[:, 3, :], (test_data.shape[0], -1))
    test_fourth_feature_n = scaler.transform(test_fourth_feature)

    # Reshape the normalized fourth feature back to 3D (samples, 1, length)
    train_fourth_feature_n = np.expand_dims(train_fourth_feature_n, axis=1)
    test_fourth_feature_n = np.expand_dims(test_fourth_feature_n, axis=1)

    # Replace the normalized fourth feature back into the original train_data and test_data
    train_data_n = np.copy(train_data)
    test_data_n = np.copy(test_data)
    train_data_n[:, 3, :] = train_fourth_feature_n[:, 0, :]
    test_data_n[:, 3, :] = test_fourth_feature_n[:, 0, :]

    return train_data_n, test_data_n

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
app_sec_rad = 5 # nautical miles
max_cutoff_range = 150 # kilometers
allowed_cutoff_error = 25 # kilometers
max_angle_change = 60 # degrees
max_fltpath_change = 20 # meters
# Preprocessing parameters
target_length = 2000 + 1
data_size = 100

required_data = int(data_size * 85)
ids_arr = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=1" % (id_tab, too_short, too_long), return_type="arrow")
ids_arr = ids_arr.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=required_data)
ADSB_arr = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_arr.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_arr = ADSB_arr.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_arr['time'] = pd.to_datetime(ADSB_arr['time'], unit='s')
ADSB_arr = ADSB_arr.groupby('flight_id')
print('Total Arrival :', ids_arr.shape[0])

required_data = int(data_size * 60)
ids_dep = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=0" % (id_tab, too_short, too_long), return_type="arrow")
ids_dep = ids_dep.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=required_data)
ADSB_dep = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_dep.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_dep = ADSB_dep.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_dep['time'] = pd.to_datetime(ADSB_dep['time'], unit='s')
ADSB_dep = ADSB_dep.groupby('flight_id')
print('Total Departure :', ids_dep.shape[0])

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
        min_z = np.min(flt_df['z'].values)
        if max_dist < (max_cutoff_range - allowed_cutoff_error) * 1000 or max_dist > max_cutoff_range * 1000 or min_z > FAF:
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
            undersampling_rate = 0.7 # Perform undersampling on southbound flights
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

    print('Rejected %d flights' % num_reject) # print the number of rejected flights
    print('Dumped %d flights' % num_dump) # print the number of dumped flights
    print('Total accepted flights :', len(df_ls)) # print the number of accepted flights
    print('Max_total_time :', np.max(total_times))
    arr_dep.append(df_ls) # arr_dep[0] is arrival, arr_dep[1] is departure

for df_ls in arr_dep:

    if len(df_ls) < data_size:
        print('Not enough data')
        continue

    random.seed(42)  # Set a common seed value for consistent sampling

    flt_traj_and_path_ls = [df.dropna().T.to_numpy() for df in df_ls if df.iloc[:, -3:].dropna(how='all').T.shape == (3, target_length-1)]
    sampled_data = random.sample(flt_traj_and_path_ls, data_size)

    flt_traj_ls = [data[:3, :] for data in sampled_data]
    flt_traj_array = np.stack(flt_traj_ls, axis=0)

    flt_path_ls = [data[-3:, :] for data in sampled_data]
    flt_path_array = np.stack(flt_path_ls, axis=0)

    n_train = int(len(flt_path_array) * 0.8)
    train_data = flt_path_array[:n_train]
    test_data = flt_path_array[n_train:]
    train_traj = flt_traj_array[:n_train]
    test_traj = flt_traj_array[n_train:]

    #train_data, test_data = normalize_fourth(train_data, test_data)

    print("Data for Arrival" if df_ls is arr_dep[0] else "Data for Departure")
    print("Flight Path Dataset Shape ====> \tTrainset: ", train_data.shape, "\tTestset: ", test_data.shape)
    print("Reference Trajectory Shape ====> \tTrainset: ", train_traj.shape, "\tTestset: ", test_traj.shape)

    ## Save signals to file
    data_dir = './data/ADSB_data_arr_test' if df_ls is arr_dep[0] else './data/ADSB_data_dep_test'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(data_dir + '/x_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(data_dir + '/x_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(data_dir + '/traj_train.pkl', 'wb') as f:
        pickle.dump(train_traj, f)
    with open(data_dir + '/traj_test.pkl', 'wb') as f:
        pickle.dump(test_traj, f)

    fig, axs = plt.subplots(6, figsize=(10, 15))
    # x plots from test_traj
    for i in range(3):
        axs[i].plot(np.arange(target_length-1), test_traj[:, i, :].T)
        axs[i].set_ylabel('X' if i == 0 else 'Y' if i == 1 else 'Z')

    # unit vector plot from test_data
    for i in range(3):
        axs[i + 3].plot(np.arange(target_length-1), test_data[:, i, :].T)
        axs[i + 3].set_ylabel('U_X' if i == 0 else 'U_Y' if i == 1 else 'U_Z')
        axs[i + 3].set_ylim(-1, 1)
    # Plot norm
    #axs[6].plot(np.arange(target_length-1), test_data[:, 3, :].T)
    #axs[6].set_ylabel('Distance between points')
    plt.tight_layout()

    # save figure
    fig.savefig(data_dir + '/sample_plot.png')

    # 2D topview plot
    fig, axs = plt.subplots(1, figsize=(10, 10))
    axs.plot(test_traj[:, 0, :].T, test_traj[:, 1, :].T)
    circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
    axs.add_patch(circle)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(data_dir + '/sample_plot_2D.png')

    # 2D topview plot with training data
    fig, axs = plt.subplots(1, figsize=(10, 10))
    axs.plot(train_traj[:, 0, :].T, train_traj[:, 1, :].T)
    circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
    axs.add_patch(circle)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(data_dir + '/sample_plot_2D_train.png')