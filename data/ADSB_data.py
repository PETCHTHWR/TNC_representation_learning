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
from atfm.utils import calculate_unit_vectors, is_smooth, calculate_bearing, check_eligible_cupy, is_flight_path_smooth, discretize_to_sectors
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
alt_col = 'baroaltitude'
max_cutoff_range = 150 # kilometers
allowed_cutoff_error = 25 # kilometers
max_angle_change = 40 # degrees
max_fltpath_change = 20 # meters
# Preprocessing parameters
target_length = 2000
data_size = 1000
r_bins, theta_bins, z_bins = 10, 24, 10

required_data = int(data_size * 120)
ids_arr = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=1" % (id_tab, too_short, too_long), return_type="arrow")
ids_arr = ids_arr.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=required_data).values.T.tolist()[0]
random.shuffle(ids_arr)
ADSB_arr = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_arr))});" % (ADSB_tab), return_type="arrow")
ADSB_arr = ADSB_arr.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_arr['time'] = pd.to_datetime(ADSB_arr['time'], unit='s')
ADSB_arr = ADSB_arr.groupby('flight_id')
print('Total Arrival :', len(ids_arr))

required_data = int(data_size * 120)
ids_dep = cx.read_sql(db_url, "SELECT DISTINCT id FROM %s WHERE ori_length>=%d AND ori_length<=%d AND arrival=0" % (id_tab, too_short, too_long), return_type="arrow")
ids_dep = ids_dep.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().sample(n=required_data).values.T.tolist()[0]
random.shuffle(ids_dep)
ADSB_dep = cx.read_sql(db_url, f"SELECT * FROM %s WHERE flight_id IN ({', '.join(map(str, ids_dep))});" % (ADSB_tab), return_type="arrow")
ADSB_dep = ADSB_dep.to_pandas(split_blocks=False, date_as_object=False).dropna()
ADSB_dep['time'] = pd.to_datetime(ADSB_dep['time'], unit='s')
ADSB_dep = ADSB_dep.groupby('flight_id')
print('Total Departure :', len(ids_dep))

arr_dep = []
for ADSB in [ADSB_arr, ADSB_dep]:
    num_reject, num_dump, num_accept = 0, 0, 0
    df_ls, bearing = [], []
    arrival = True if ADSB is ADSB_arr else False
    bins = [0, 110, 150, 210, 360]
    counts = {i: 0 for i in range(4)}

    for id, flt_df in tqdm(ADSB, total=len(ADSB), desc="Processing groups", ncols=100):

        # check if the flight is eligible
        flt_df = flt_df.set_index('time')  # set time as index
        if check_eligible_cupy(flt_df, min_alt_change, FAF, icn_lat, icn_lon, app_sec_rad, alt_col=alt_col, max_range=max_cutoff_range):
            try:
                flt_df = preprocess(flt_df.dropna(), ref_lat=icn_lat, ref_lon=icn_lon, ref_alt=icn_alt, periods=target_length,
                                    max_range=max_cutoff_range, alt_column=alt_col, freq='1s', zscore_threshold=3, window_size=3, order=2)  # preprocess
            except Exception:  # if preprocessing fails
                num_reject += 1
                continue
        else:
            num_reject += 1
            continue

        # Undersampling
        north_bearing = calculate_bearing(flt_df.iloc[-1]['x'], flt_df.iloc[-1]['y']) \
            if not arrival else calculate_bearing(flt_df.iloc[0]['x'], flt_df.iloc[0]['y'])
        region = np.digitize(north_bearing, bins) - 1  # Determine the region of the bearing
        if counts[region] < data_size / 4:  # Check if the region has not exceeded its count
            counts[region] += 1
        else:
            num_dump += 1
            continue

        # Check if the flight is really eligible if not rejected and subtract the count
        flt_arr = cp.asarray(flt_df[['x', 'y', 'z']].values)  # Convert DataFrame to CuPy array
        max_dist = cp.max(cp.linalg.norm(flt_arr[:, :2], axis=1))  # Calculate maximum Euclidean norm for x, y
        min_alt = cp.min(flt_arr[:, 2])  # Calculate minimum altitude
        if max_dist < (max_cutoff_range - allowed_cutoff_error) * 1000 \
                or max_dist > max_cutoff_range * 1000 or min_alt > FAF:
            counts[region] -= 1
            num_reject += 1
            continue

        # Check if the flight is smooth and subtract the count if not
        unit_df = calculate_unit_vectors(flt_df) # Calculate unit vectors
        if not is_smooth(unit_df, max_angle_change) \
                or not is_flight_path_smooth(unit_df, max_fltpath_change) \
                or unit_df.shape[0] != target_length - 1:
            counts[region] -= 1
            num_reject += 1
            continue

        # Discretize the unit vectors and subtract the count if not
        disc_df = discretize_to_sectors(flt_df, r_bins=r_bins, theta_bins=theta_bins, z_bins=z_bins, r_max = max_cutoff_range * 1000)
        joined_df = pd.concat([flt_df, unit_df, disc_df], axis=1) # join the dataframes
        joined_df[unit_df.columns.tolist()] = joined_df[unit_df.columns.tolist()].shift(-1) # shift the unit vectors by 1
        for col in unit_df.columns.tolist():
            joined_df.loc[joined_df.index[-1], col] = joined_df.loc[joined_df.index[-2], col]
        if joined_df.dropna().shape[0] != target_length:
            counts[region] -= 1
            num_reject += 1
            continue

        df_ls.append(joined_df.dropna())  # append the dataframe
        bearing.append(north_bearing)  # append the bearing
        num_accept += 1 # Stop if the number of accepted flights reaches the data size
        if num_accept == data_size:
            break

    print('Rejected %d flights' % num_reject) # print the number of rejected flights
    print('Dumped %d flights' % num_dump) # print the number of dumped flights
    print('Total accepted flights :', len(df_ls)) # print the number of accepted flights
    arr_dep.append(df_ls) # arr_dep[0] is arrival, arr_dep[1] is departure

    # Plot the histogram of the bearing to confirm that the undersampling is done correctly
    fig, ax = plt.subplots()
    bearing = np.array(bearing)
    groups = ["0-110", "110-150", "150-210", "210-360"]
    grouped_bearings = [np.digitize(b, bins=[0, 110, 150, 210, 360]) - 1 for b in bearing]
    counts, bins, patches = ax.hist(grouped_bearings, bins=np.arange(5) - 0.5, edgecolor='black')
    ax.set_xticks(range(4))
    ax.set_xticklabels(groups)
    ax.set_xlabel('Bearing Group (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of North Bearing Groups')
    path = './data/ADSB_data_arr_test' if arrival else './data/ADSB_data_dep_test'
    plt.savefig(f'{path}/bearing_hist.png')

for df_ls in arr_dep:

    if len(df_ls) < data_size:
        print('Not enough data')
        continue

    random.seed(42)  # Set a common seed value for consistent sampling

    flt_traj_and_path_ls = [df.T.to_numpy() for df in df_ls if df.shape[0] == target_length]
    sampled_data = random.sample(flt_traj_and_path_ls, data_size)
    flt_traj_ls = [data[:3, :] for data in sampled_data]
    flt_traj_array = np.stack(flt_traj_ls, axis=0)

    flt_path_ls = [data[3:, :] for data in sampled_data]
    flt_path_array = np.stack(flt_path_ls, axis=0)

    n_train = int(len(flt_path_array) * 0.8)
    train_data = flt_path_array[:n_train]
    test_data = flt_path_array[n_train:]
    train_data, test_data = normalize_fourth(train_data, test_data)

    # Reference Trajectory
    train_traj = flt_traj_array[:n_train]
    test_traj = flt_traj_array[n_train:]

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

    # Plot the trajectory profile
    fig, axs = plt.subplots(test_traj.shape[1], figsize=(10, 10))
    for i in range(test_traj.shape[1]):
        axs[i].plot(np.arange(target_length), test_traj[:, i, :].T)
        axs[i].set_ylabel('X' if i == 0 else 'Y' if i == 1 else 'Z')
    plt.tight_layout()
    fig.savefig(data_dir + '/original_trajectory_profile.png')

    # Plot the states profile
    fig, axs = plt.subplots(test_data.shape[1], figsize=(10, 10))
    label = ['u_x', 'u_y', 'u_z', 'dr', 'sec_r', 'sec_theta', 'sec_z']
    for i in range(test_data.shape[1]):
        axs[i].plot(np.arange(target_length), test_data[:, i, :].T)
        axs[i].set_ylabel(label[i])
        if i != 3:
            axs[i].set_ylim(-1, 1)
    plt.tight_layout()
    fig.savefig(data_dir + '/states_profile.png')

    # Set up your sectors' edges
    r_max = max_cutoff_range * 1000
    theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)
    r_edges = np.linspace(0, r_max, r_bins + 1)

    # 2D topview plot
    fig, axs = plt.subplots(1, figsize=(10, 10))
    axs.plot(test_traj[:, 0, :].T, test_traj[:, 1, :].T)

    # Draw your sectors
    for r in r_edges:
        circle = plt.Circle((0, 0), r, color='black', fill=False)
        axs.add_artist(circle)

    for theta in theta_edges:
        axs.plot([0, r_max * np.cos(theta)], [0, r_max * np.sin(theta)], color='black')

    # Add an outer circle
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

    # Draw your sectors
    for r in r_edges:
        circle = plt.Circle((0, 0), r, color='black', fill=False)
        axs.add_artist(circle)

    for theta in theta_edges:
        axs.plot([0, r_max * np.cos(theta)], [0, r_max * np.sin(theta)], color='black')

    # Add an outer circle
    circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
    axs.add_patch(circle)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(data_dir + '/sample_plot_2D_train.png')