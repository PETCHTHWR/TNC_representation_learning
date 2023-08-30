import pandas as pd
import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import connectorx as cx
from atfm.preprocess import preprocess, smooth_savgol_vel, remove_outliers_vel
from atfm.utils import calculate_unit_vectors, is_smooth, calculate_bearing, check_eligible_cupy, is_flight_path_smooth, discretize_to_sectors, calculate_velocities
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

def get_data(data_size, multiplier, db_url, id_tab, ADSB_tab, too_short, too_long, arrival_flag, alt_col):
    required_data = int(data_size * multiplier / 4)
    ids = []
    for sec_num in range(1, 5):
        query_ids = f"SELECT DISTINCT id FROM {id_tab} WHERE ori_length>={too_short} AND ori_length<={too_long} AND arrival={arrival_flag} AND bearing_sector={sec_num} ORDER BY RAND() LIMIT {required_data}"
        ids_q = cx.read_sql(db_url, query_ids, return_type="arrow")
        ids_q = ids_q.to_pandas(split_blocks=False, date_as_object=False).dropna().drop_duplicates().values.T.tolist()[0]
        ids.extend(ids_q)
    random.shuffle(ids)
    query_ADSB = f"SELECT time, flight_id, lat, lon, {alt_col} FROM {ADSB_tab} WHERE flight_id IN ({', '.join(map(str, ids))});"
    ADSB = cx.read_sql(db_url, query_ADSB, return_type="arrow")
    ADSB = ADSB.to_pandas(split_blocks=False, date_as_object=False).dropna()
    ADSB['time'] = pd.to_datetime(ADSB['time'], unit='s')
    ADSB = ADSB.groupby('flight_id')
    print('Total data :', len(ids))
    return ADSB

def normalize_features(train_data, test_data, start_idx, end_idx):
    """
    Normalize features in the train and test data arrays using MinMaxScaler

    Parameters:
    train_data (numpy.ndarray): The training data to normalize
    test_data (numpy.ndarray): The testing data to normalize
    start_idx (int): The starting index of features to normalize
    end_idx (int): The ending index of features to normalize

    Returns:
    tuple: The normalized training and testing data
    """

    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Initialize normalized train and test data as copies of the original data
    train_data_n = np.copy(train_data)
    test_data_n = np.copy(test_data)

    # Loop over the features from start_idx to end_idx
    for i in range(start_idx, end_idx + 1):
        # Reshape the feature of train data to 2D (samples, length)
        train_feature = np.reshape(train_data[:, i, :], (train_data.shape[0], -1))

        # Fit the scaler on the train feature and normalize it
        scaler.fit(train_feature)
        train_feature_n = scaler.transform(train_feature)

        # Normalize the feature of the test data using the fitted scaler
        test_feature = np.reshape(test_data[:, i, :], (test_data.shape[0], -1))
        test_feature_n = scaler.transform(test_feature)

        # Reshape the normalized feature back to 3D (samples, 1, length)
        train_feature_n = np.expand_dims(train_feature_n, axis=1)
        test_feature_n = np.expand_dims(test_feature_n, axis=1)

        # Replace the normalized feature back into the original train_data_n and test_data_n
        train_data_n[:, i, :] = train_feature_n[:, 0, :]
        test_data_n[:, i, :] = test_feature_n[:, 0, :]

    return train_data_n, test_data_n

def normalize_velocity(train_data, test_data, start_idx, end_idx):
    """
    Normalize velocity features in the train and test data arrays using MinMaxScaler

    Parameters:
    train_data (numpy.ndarray): The training data to normalize
    test_data (numpy.ndarray): The testing data to normalize
    start_idx (int): The starting index of velocity features to normalize
    end_idx (int): The ending index of velocity features to normalize

    Returns:
    tuple: The normalized training and testing data
    """

    # Create a MinMaxScaler object
    scaler = MaxAbsScaler()

    # Create an imputer
    imputer = SimpleImputer(strategy='constant', fill_value=-1)

    # Stack the velocity features from the train data and fit the scaler
    velocity_stack = np.hstack([train_data[:, i, :].flatten() for i in range(start_idx, end_idx + 1)] +
                                     [test_data[:, i, :].flatten() for i in range(start_idx, end_idx + 1)]).reshape(-1, 1)
    velocity_stack_imp = imputer.fit_transform(velocity_stack)

    scaler.fit(velocity_stack_imp)

    # Initialize normalized train and test data as copies of the original data
    train_data_n = np.copy(train_data)
    test_data_n = np.copy(test_data)

    # Normalize the velocities of the current axis in the train and test data using the fitted scaler and imputer
    for i in range(start_idx, end_idx + 1):
        train_data_n[:, i, :] = scaler.transform(imputer.transform(train_data[:, i, :].flatten().reshape(-1, 1))).reshape(train_data[:, i, :].shape)
        test_data_n[:, i, :] = scaler.transform(imputer.transform(test_data[:, i, :].flatten().reshape(-1, 1))).reshape(test_data[:, i, :].shape)

    # Replace -1 back with NaN
    train_data_n[:, start_idx:end_idx + 1, :][train_data_n[:, start_idx:end_idx + 1, :] == -1] = np.nan
    test_data_n[:, start_idx:end_idx + 1, :][test_data_n[:, start_idx:end_idx + 1, :] == -1] = np.nan

    return train_data_n, test_data_n

def standardize_data(train_data, test_data):
    """
    Standardize the data based on the training data mean and standard deviation.
    """
    num_features = train_data.shape[1]

    # Placeholder for standardized data
    standardized_train = np.zeros_like(train_data)
    standardized_test = np.zeros_like(test_data)

    for i in range(num_features):
        # Reshape each feature across all samples and timesteps for training data
        feature_train = train_data[:, i, :].reshape(-1)
        feature_test = test_data[:, i, :].reshape(-1)

        # Standardize
        mean = np.mean(feature_train)
        std = np.std(feature_train)
        standardized_train_feature = (feature_train - mean) / (std + 1e-10)
        standardized_test_feature = (feature_test - mean) / (std + 1e-10)

        # Reshape back and populate the standardized data
        standardized_train[:, i, :] = standardized_train_feature.reshape(train_data.shape[0], -1)
        standardized_test[:, i, :] = standardized_test_feature.reshape(test_data.shape[0], -1)

    return standardized_train, standardized_test

# Query parameters
db_url = 'mysql://lics:aelics070@143.248.69.46:13306/atfm_new'
id_tab, ADSB_tab = 'flight', 'trajectory'
too_short, too_long = 200, 3500
req_mul_arr, req_mul_dep = 1.04, 1.04
data_size = 5000

# Filtering parameters
icn_lat, icn_lon, icn_alt = 37.463333, 126.440002, 8.0 # degrees, degrees, meters
min_alt_change, FAF, app_sec_rad = 610, 480, 5 # meters
max_cutoff_range, allowed_cutoff_error = 150, 25 # kilometers
alt_col = 'baroaltitude' # column name for altitude
max_angle_change, max_fltpath_change = 40, 20 # degrees

# Preprocessing parameters
target_length = 2000
r_bins, theta_bins, x_bins, y_bins, z_bins = 10, 12, 10, 10, 10
unify = True

#ADSB_arr = get_data(data_size, req_mul_arr, db_url, id_tab, ADSB_tab, too_short, too_long, 1)
#ADSB_dep = get_data(data_size, req_mul_dep, db_url, id_tab, ADSB_tab, too_short, too_long, 0)
#for ADSB in [ADSB_arr, ADSB_dep]:

arr_dep = []
for arrival in [True, False]:
    req = req_mul_arr if arrival else req_mul_dep
    ADSB = get_data(data_size, req, db_url, id_tab, ADSB_tab, too_short, too_long, int(arrival), alt_col)
    num_reject, num_dump, num_accept = 0, 0, 0
    df_ls, bearing, bins = [], [], [0, 110, 150, 210, 360]
    counts = {i: 0 for i in range(4)}

    for id, flt_df in tqdm(ADSB, total=len(ADSB), desc="Processing ADSB", ncols=100):

        # check if the flight is eligible
        flt_df = flt_df.set_index('time')  # set time as index
        if check_eligible_cupy(flt_df, min_alt_change, FAF, icn_lat, icn_lon, app_sec_rad, alt_col=alt_col, max_range=max_cutoff_range):
            try:
                flt_df = preprocess(flt_df.dropna(), ref_lat=icn_lat, ref_lon=icn_lon, ref_alt=icn_alt,
                                    periods=target_length, max_range=max_cutoff_range, alt_column=alt_col, unify=unify,
                                    freq='1s', zscore_threshold=3, window_size=5, order=2)  # preprocess
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
        touch_alt = flt_arr[-1, 2] if arrival else flt_arr[0, 2]  # Calculate touch altitude
        if max_dist < (max_cutoff_range - allowed_cutoff_error) * 1000 \
                or max_dist > max_cutoff_range * 1000 or min_alt > FAF or touch_alt > FAF:
            counts[region] -= 1
            num_reject += 1
            continue

        # Check if the flight is smooth and subtract the count if not
        unit_df = calculate_unit_vectors(flt_df) # Calculate unit vectors
        if not is_smooth(unit_df, max_angle_change) \
                or not is_flight_path_smooth(unit_df, max_fltpath_change) \
                or unit_df.shape[0] != flt_df.shape[0] - 1:
            counts[region] -= 1
            num_reject += 1
            continue

        # Discretize the data
        disc_df = discretize_to_sectors(flt_df, r_bins=r_bins, theta_bins=theta_bins,
                                        x_bins=x_bins, y_bins=y_bins, z_bins=z_bins,
                                        r_max = max_cutoff_range * 1000)

        # Join the dataframes
        joined_df = pd.concat([flt_df, unit_df, disc_df[['r_sector', 'theta_sector', 'x_sector', 'y_sector', 'z_sector']]], axis=1) # join the dataframes
        joined_df[unit_df.columns.tolist()] = \
            joined_df[unit_df.columns.tolist()].shift(-1) # shift the unit vectors by 1
        joined_df = joined_df.reset_index(drop=True)
        for col in unit_df.columns.tolist():
            joined_df.loc[joined_df.index[-1], col] = joined_df.loc[joined_df.index[-2], col]
        if joined_df.dropna().shape[0] != flt_df.shape[0]:
            counts[region] -= 1
            num_reject += 1
            continue

        df_ls.append(joined_df.dropna())
        # append the dataframe
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
    path = './data/ADSB_data_arr' if arrival else './data/ADSB_data_dep'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(f'{path}/bearing_hist.png')

for df_ls in arr_dep:

    if len(df_ls) < data_size:
        print('Not enough data')
        continue

    random.seed(42)  # Set a common seed value for consistent sampling

    max_length = max(len(df) for df in df_ls)
    df_ls_reidx = [df.reindex(range(max_length), fill_value=np.nan) for df in df_ls]
    flt_traj_and_path_ls = [df.T.to_numpy() for df in df_ls_reidx if df.shape[0] == max_length]

    sampled_data = random.sample(flt_traj_and_path_ls, data_size)
    flt_traj_ls = [data[:3, :] for data in sampled_data]
    flt_traj_array = np.stack(flt_traj_ls, axis=0)
    print(flt_traj_array.shape)
    flt_path_ls = [data[3:, :] for data in sampled_data]
    flt_path_array = np.stack(flt_path_ls, axis=0)
    print(flt_path_array.shape)

    #flt_path_array = np.concatenate((flt_path_array,
    #                                 flt_traj_array[:, :2, :]/max_cutoff_range/1000,
    #                                 flt_traj_array[:, 2:, :]/max_cutoff_range/100*2-1), axis=1)

    n_train = int(len(flt_path_array) * 0.5)
    train_data, test_data = standardize_data(flt_path_array[:n_train], flt_path_array[n_train:])
    train_data[:, 2, :] = np.clip(train_data[:, 2, :], -3, 3)
    test_data[:, 2, :] = np.clip(test_data[:, 2, :], -3, 3)

    train_traj, test_traj = flt_traj_array[:n_train], flt_traj_array[n_train:]

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

    # Plot the trajectory profile
    fig, axs = plt.subplots(test_traj.shape[1], figsize=(10, 10))
    for i in range(test_traj.shape[1]):
        axs[i].plot(np.arange(max_length), test_traj[:, i, :].T, alpha=0.5)
        axs[i].set_ylabel('X' if i == 0 else 'Y' if i == 1 else 'Z')
    plt.tight_layout()
    fig.savefig(data_dir + '/original_trajectory_profile.png')

    # Plot the states profile
    fig, axs = plt.subplots(test_data.shape[1], figsize=(10, 20))
    label = ['u_x', 'u_y', 'u_z', 'r', 'theta','x', 'y', 'z']
    for i in range(test_data.shape[1]):
        axs[i].plot(np.arange(max_length), test_data[:, i, :].T, alpha=0.5)
        axs[i].set_ylabel(label[i])
        #axs[i].set_ylim(-1, 1)
    plt.tight_layout()
    fig.savefig(data_dir + '/states_profile.png')

    # Plot the trajectory profile
    fig, axs = plt.subplots(train_traj.shape[1], figsize=(10, 10))
    for i in range(train_traj.shape[1]):
        axs[i].plot(np.arange(max_length), train_traj[:, i, :].T, alpha=0.5)
        axs[i].set_ylabel('X' if i == 0 else 'Y' if i == 1 else 'Z')
    plt.tight_layout()
    fig.savefig(data_dir + '/original_trajectory_profile_train.png')

    # Plot the states profile
    fig, axs = plt.subplots(train_data.shape[1], figsize=(10, 20))
    label = ['u_x', 'u_y', 'u_z', 'r', 'theta','x', 'y', 'z']
    for i in range(train_data.shape[1]):
        axs[i].plot(np.arange(max_length), train_data[:, i, :].T, alpha=0.5)
        axs[i].set_ylabel(label[i])
        #axs[i].set_ylim(-1, 1)
    plt.tight_layout()
    fig.savefig(data_dir + '/states_profile_train.png')

    # Set up your sectors' edges
    r_max = max_cutoff_range * 1000
    theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)
    r_edges = np.linspace(0, r_max, r_bins + 1)

    # 2D topview plot
    fig, axs = plt.subplots(1, figsize=(10, 10))
    axs.plot(test_traj[:, 0, :].T, test_traj[:, 1, :].T, alpha=0.5)

    # Draw your sectors
    for r in r_edges:
        circle = plt.Circle((0, 0), r, color='black', fill=False, alpha=0.2)
        axs.add_artist(circle)

    for theta in theta_edges:
        axs.plot([0, r_max * np.cos(theta)], [0, r_max * np.sin(theta)], color='black', alpha=0.2)

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
    axs.plot(train_traj[:, 0, :].T, train_traj[:, 1, :].T, alpha=0.5)

    # Draw your sectors
    for r in r_edges:
        circle = plt.Circle((0, 0), r, color='black', fill=False, alpha=0.2)
        axs.add_artist(circle)

    for theta in theta_edges:
        axs.plot([0, r_max * np.cos(theta)], [0, r_max * np.sin(theta)], color='black', alpha=0.2)

    # Add an outer circle
    circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
    axs.add_patch(circle)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.grid(True)
    axs.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(data_dir + '/sample_plot_2D_train.png')