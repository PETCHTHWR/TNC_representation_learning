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
from atfm.utils import check_eligible, calculate_unit_vectors, is_smooth, calculate_bearing
from sklearn.preprocessing import MinMaxScaler

def normalize_fourth(train_data, test_data):
    """Normalize only the fourth feature in the train and test data arrays using MinMaxScaler"""

    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

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
sample_size = 55000

# Filtering parameters
icn_lat, icn_lon, icn_alt = 37.49491667, 126.43033333, 8.0
min_alt_change = 2000 / 3.281 # meters
min_FAF_baro = 1600 / 3.281 # meters
app_sector_rad = 25 # nautical miles
max_cutoff_range = 150 # kilometers
undersampling_rate = 0.7
# Preprocessing parameters
target_length = 2000 + 1
data_size = 2000

ids_arr = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=1" % (id_tab, too_short, too_long), return_type="arrow")
ids_arr = ids_arr.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=sample_size)
ADSB_arr = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_arr.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_arr = ADSB_arr.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_arr['time'] = pd.to_datetime(ADSB_arr['time'], unit='s')
print('Total Arrival :', ids_arr.shape[0])

ids_dep = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=0" % (id_tab, too_short, too_long), return_type="arrow")
ids_dep = ids_dep.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=sample_size)
ADSB_dep = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_dep.values.T.tolist()[0]))});" % (ADSB_tab), return_type="arrow")
ADSB_dep = ADSB_dep.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_dep['time'] = pd.to_datetime(ADSB_dep['time'], unit='s')
print('Total Departure :', ids_dep.shape[0])


arr_dep = []
for ADSB in [ADSB_arr, ADSB_dep]:
    num_reject = 0
    df_ls = []
    arrival = True if ADSB is ADSB_arr else False

    for id in tqdm(set(ADSB['flight_id'].values.tolist())):

        flt_df = ADSB.loc[ADSB['flight_id'] == id]
        flt_df = flt_df.set_index('time')

        if not check_eligible(flt_df, min_alt_change, min_FAF_baro, icn_lat, icn_lon, app_sector_rad, alt_col='baroaltitude', max_range=max_cutoff_range):
            num_reject += 1
            continue

        try:
            flt_df = flt_df.dropna()
            flt_df = preprocess(flt_df, ref_lat=icn_lat, ref_lon=icn_lon, ref_alt=icn_alt, periods=target_length,
                                max_range_x=max_cutoff_range, alt_column='baroaltitude')
        except Exception:
            num_reject += 1
            continue

        if np.max(np.linalg.norm(flt_df[['x', 'y']].values, axis=1)) < (max_cutoff_range * 1000) - 500:
            num_reject += 1
            continue

        if arrival: # Perform undersampling
            if calculate_bearing(flt_df.iloc[0]['x'], flt_df.iloc[0]['y']) > 120 or calculate_bearing(flt_df.iloc[0]['x'], flt_df.iloc[0]['y']) < 300:
                if np.random.choice([True, False], p=[undersampling_rate, 1-undersampling_rate]):
                    num_reject += 1
                    continue
        else:
            if calculate_bearing(flt_df.iloc[-1]['x'], flt_df.iloc[-1]['y']) > 240 and calculate_bearing(flt_df.iloc[-1]['x'], flt_df.iloc[-1]['y']) < 300:
                if np.random.choice([True, False], p=[undersampling_rate, 1-undersampling_rate]):
                    num_reject += 1
                    continue

        unit_df = calculate_unit_vectors(flt_df)
        if not is_smooth(unit_df, 60):
            num_reject += 1
            continue

        joined_df = pd.concat([flt_df, unit_df], axis=1)
        joined_df[unit_df.columns] = joined_df[unit_df.columns].shift(-1)
        df_ls.append(joined_df)

    print('Rejected %d flights' % num_reject)
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
    data_dir = './data/ADSB_data_arr' if df_ls is arr_dep[0] else './data/ADSB_data_dep'
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

    # Plot norm
    #axs[6].plot(np.arange(target_length-1), test_data[:, 3, :].T)
    #axs[6].set_ylabel('Distance between points')
    plt.tight_layout()

    # save figure
    fig.savefig(data_dir + '/sample_plot.png')

    # 2D topview plot
    fig, axs = plt.subplots(1, figsize=(10, 8))
    axs.plot(test_traj[:, 0, :].T, test_traj[:, 1, :].T)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(data_dir + '/sample_plot_2D.png')