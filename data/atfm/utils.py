import numpy as np
import ciso8601
# from numba import jit
from geopy import distance
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import pymap3d as pm
from functools import wraps
from time import time
import cupy as cp
import pandas as pd


def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """
    Convert (lat, lon) to (x, y)
    :param lat: latitude
    :param lon: longitude
    :param ref_lat: reference latitude
    :param ref_lon: reference longitude
    :return: (x, y)
    """
    x, y, _ = pm.geodetic2enu(lat, lon, 0, ref_lat, ref_lon, 0)
    return x, y

def latlonalt_to_xyz(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert (lat, lon) to (x, y)
    :param lat: latitude
    :param lon: longitude
    :param alt: altitude (meters)
    :param ref_lat: reference latitude
    :param ref_lon: reference longitude
    :param ref_alt: reference altitude (meters)
    :return: (x, y)
    """
    x, y, z = pm.geodetic2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
    return x, y, z

def time2stamp(t):
    return int(ciso8601.parse_datetime(t).timestamp())


def data_resampling(df, num_row):
    """
    :param df: pd.Dataframe; intended to resampled
    :param num_row: int; number of time grid to be use for resampled
    :return df: pd.Dataframe; resampled dataframe
    """
    resampled_time = np.linspace(df.index.min(), df.index.max(), num_row)  # Create resampled time grid
    df = df.reindex(df.index.union(resampled_time)).interpolate(method='linear').reindex(
        resampled_time)  # Add resampled grid to df --> interpolate --> reindex
    return df


def flat_list(df):
    """
    :param df: pd.Dataframe; dataframe intended to be flattened
    :return flat_list: list shape = (len(df),); flattened list containing dataframe's rows
    """
    # flatten the dataframe and include index
    flat_list = [j for i in df.reset_index().values for j in i]
    return flat_list


def filter_squawk_repeating(df):
    """
    :param df: raw trajectory data in form pd.Dataframe
    :return: the filtered flt trajectory sacrificed the part with repeating squawk

    FROM ORIGINAL FUNCTION
    try: # Check if squawk has been reuse or not; the state vectors those are updated beyond 2 hours from the first time update is sacrificed and considered as another trajectory.
        flt_df = flt_df[flt_df['hour'] <= flt_df['hour'].iloc[0] + 3600 * 2]
    except IndexError: # Means that flt_df is empty --> skip
        num_rejected += 1
        continue
    """

    if df.empty:
        return df
    else:
        df = df[df['hour'] <= df['hour'].iloc[
            0] + 3600 * 2]  # Filter for only the first two hours to save computational time
        time_diff = df['time'].diff().fillna(value=0)  # Calculate the time difference between successive rows
        idx = (time_diff > 90).idxmax()

        if idx == 0 or idx == None:
            return df  # Find the index of the first row with time difference greater than 90
        else:
            return df.loc[:idx - 1]  # Select the rows before the first row with time difference greater than 90


def check_eligible(df, min_alt_change, min_FAF_baro, ad_lat, ad_lon, app_sector_rad, alt_col='geoaltitude', max_range=100):
    """
    :param df:
    :param min_FAF_baro:
    :param ad_lat:
    :param ad_lon:
    :param app_sector_rad:
    :param min_traj_length:
    :param max_traj_length:
    :return:

    FROM ORIGINAL FUNCTION
    if flt_df.empty:
        num_rejected += 1
        continue

    # If the minimum pressure altitude is above FAF, the aircraft is not intended to land or depart from this airport
    if flt_df['baroaltitude'].min() > min_FAF_baro:
        num_rejected += 1
        continue

    # If the lateral position at the lowest altitude is not within 10NM, the trajectory is not in the approach sector
    min_alt_row = flt_df.loc[flt_df['geoaltitude'].idxmin()] # The row at minimum altitude
    min_alt_dis = distance.great_circle(aerodrome_loc, (min_alt_row['lat'], min_alt_row['lon'])).nm # Use geopy to calculate great circle distance
    if min_alt_dis > app_sector_rad:
        num_rejected += 1
        continue

    # The meaningful trajectory should have a certain lengths
    if flt_df.shape[0] < min_traj_length or flt_df.shape[0] > max_traj_length:
        num_rejected += 1
        continue
    """

    if df.empty:
        return False
    elif df.iloc[0][alt_col] == df.iloc[-1][alt_col]:
        return False
    elif df[alt_col].max() - df[alt_col].min() < min_alt_change:
        return False
    elif df[alt_col].min() > min_FAF_baro:
        return False
    elif df[alt_col].max() < min_FAF_baro:
        return False
    elif distance.great_circle((ad_lat, ad_lon), (
            df.loc[df[alt_col].idxmin()]['lat'], df.loc[df[alt_col].idxmin()]['lon'])).nm > app_sector_rad:
        return False
    elif distance.great_circle((ad_lat, ad_lon),
            (df.iloc[0]['lat'], df.iloc[0]['lon'])).km < max_range \
            and distance.great_circle((ad_lat, ad_lon),
            (df.iloc[-1]['lat'], df.iloc[-1]['lon'])).km < max_range:
        return False
    else:
        return True

def check_eligible_cupy(df, min_alt_change, min_FAF_baro, ad_lat, ad_lon, app_sector_rad, alt_col='geoaltitude', max_range=100):
    df_array = cp.asarray(df)
    alt_col_index = df.columns.get_loc(alt_col)
    lat_col_index = df.columns.get_loc('lat')
    lon_col_index = df.columns.get_loc('lon')

    if df_array.size == 0:
        return False
    elif df_array[0, alt_col_index] == df_array[-1, alt_col_index]:
        return False
    elif cp.amax(df_array[:, alt_col_index]) - cp.amin(df_array[:, alt_col_index]) < min_alt_change:
        return False
    elif cp.amin(df_array[:, alt_col_index]) > min_FAF_baro:
        return False
    elif cp.amax(df_array[:, alt_col_index]) < min_FAF_baro:
        return False
    elif distance.great_circle((ad_lat, ad_lon),
                               (df_array[cp.argmin(df_array[:, alt_col_index]), lat_col_index], df_array[cp.argmin(df_array[:, alt_col_index]), lon_col_index])).nm > app_sector_rad:
        return False
    elif distance.great_circle((ad_lat, ad_lon), (df_array[0, lat_col_index], df_array[0, lon_col_index])).km < max_range \
            and distance.great_circle((ad_lat, ad_lon), (df_array[-1, lat_col_index], df_array[-1, lon_col_index])).km < max_range:
        return False
    else:
        return True

def traj_unpack(traj):
    """
    TODO: write unpacking function
    :param traj: list or np.array in the traj format
    :return traj_df: pd.Dataframe; dataframe of trajectory unpacked from list
    """
    traj_df = 0
    return traj_df


def map_RWY(flt_df, RWY_df, arrival):
    if arrival == True:
        last_lat, last_long = flt_df.iloc[-1]['lat'], flt_df.iloc[-1]['lon']
        distance_list = [distance.great_circle((last_lat, last_long), (RWY.lat, RWY.long)).nm for RWY in RWY_df.itertuples(index=False)]
        RWY_index = distance_list.index(min(distance_list))
        return RWY_df.iloc[RWY_index]['RWY']
    else:
        first_lat, first_long = flt_df.iloc[0]['lat'], flt_df.iloc[0]['lon']
        distance_list = [distance.great_circle((first_lat, first_long), (RWY.lat, RWY.long)).nm for RWY in RWY_df.itertuples(index=False)]
        RWY_index = distance_list.index(min(distance_list))
        return RWY_df.iloc[RWY_index]['Opposite']

def map_corridors(flt_df, SID_df, STAR_df, arrival):
    if arrival == True:
        first_lat, first_long = flt_df.iloc[0]['lat'], flt_df.iloc[0]['lon']
        distance_list = [distance.great_circle((first_lat, first_long), (WPT.lat, WPT.long)).nm for WPT in STAR_df.itertuples(index=False)]
        WPT_index = distance_list.index(min(distance_list))
        return STAR_df.iloc[WPT_index]['Corridor']
    else:
        last_lat, last_long = flt_df.iloc[-1]['lat'], flt_df.iloc[-1]['lon']
        distance_list = [distance.great_circle((last_lat, last_long), (WPT.lat, WPT.long)).nm for WPT in SID_df.itertuples(index=False)]
        WPT_idex = distance_list.index(min(distance_list))
        return SID_df.iloc[WPT_idex]['Corridor']

def multiprocessing(fun, args, num_thread, tqdm_enable=True):
    """
    :param fun: function; function to be parallelized
    :param args: list; list of arguments to be passed to the function
    :param num_thread: int; number of thread to be used
    :return result: list; list of results from each thread
    """
    with Pool(num_thread) as p:
        if tqdm_enable:
            result = list(tqdm(p.imap(fun, args), total=len(args)))
        else:
            result = list(p.imap(fun, args))
    return result


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def calculate_unit_vectors(flt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the unit vectors of the trajectory DataFrame.
    Input:
        flt_df: A DataFrame containing the trajectory data of a single flight.
    Output:
        unit_df: A DataFrame containing the unit vectors of the trajectory.
    """
    diff_df = flt_df.diff().dropna() # Create a new DataFrame for the differences
    norms = np.sqrt((diff_df**2).sum(axis=1)) # Calculate the norm of each difference vector
    unit_df = diff_df.div(norms, axis=0) # Divide each difference vector by its norm to get the unit vectors
    unit_df.columns = ['u_x', 'u_y', 'u_z'] # The unit vectors are now stored in unit_df as 'x', 'y', and 'z'
    # unit_df['r'] = norms # Add the norm of the difference vectors as a column

    return unit_df

def calculate_velocities(flt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the velocities of the trajectory DataFrame.
    @param flt_df: A DataFrame containing the trajectory data of a single flight.
    @return: A DataFrame containing the velocities of the trajectory.
    """
    # Check if all the necessary columns exist in the DataFrame
    for column in ['x', 'y', 'z']:
        if column not in flt_df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Calculate the time difference between each point
    time_diff = flt_df.index.to_series().diff().dt.total_seconds().values[1:]

    # Calculate the differences between each point
    vel_df = flt_df.diff().iloc[1:]

    # Calculate the velocity in each direction
    for column in ['x', 'y', 'z']:
        vel_df[f'v_{column}'] = vel_df[column] / time_diff

    # Drop the original position columns
    vel_df.drop(['x', 'y', 'z'], axis=1, inplace=True)

    return vel_df

def discretize_to_sectors(df: pd.DataFrame, r_bins: int, theta_bins: int, x_bins: int, y_bins: int, z_bins: int, r_max: float):
    """
    Discretize 3D trajectories into radial, angular and vertical sectors.
    The computed radius and bearing values are normalized to the range [-1, 1].
    Input:
        df: A DataFrame containing the 3D trajectories.
        r_bins: The number of radial sectors.
        theta_bins: The number of angular sectors.
        z_bins: The number of altitude sectors.
        r_max: The maximum radius to consider.
    Output:
        df_sectors: A DataFrame with the sector assignments and normalized radius and bearing values.
    """
    # Calculate radius and bearing from x, y, z
    radius = (df['x']**2 + df['y']**2)**0.5
    bearing = np.arctan2(df['x'], df['y'])

    # Normalize bearing to 0-2pi and then to -1 to 1 range
    bearing = (bearing + 2 * np.pi) % (2 * np.pi)
    normalized_theta = 2 * (bearing / (2 * np.pi)) - 1

    # Normalize radius to -1 to 1 range
    normalized_radius = 2 * (radius / r_max) - 1

    # Discretize into sectors and normalize to -1 to 1
    r_sector = pd.cut(radius, bins=np.linspace(0, r_max, r_bins + 1), labels=False, include_lowest=True)
    r_sector = 2 * (r_sector / r_bins) - 1
    theta_sector = pd.cut(bearing, bins=np.linspace(0, 2 * np.pi, theta_bins + 1), labels=False, include_lowest=True)
    theta_sector = 2 * (theta_sector / theta_bins) - 1

    x_sector = pd.cut(df['x'], bins=np.linspace(-r_max, r_max, x_bins + 1), labels=False, include_lowest=True)
    x_sector = 2 * (x_sector / x_bins) - 1
    y_sector = pd.cut(df['y'], bins=np.linspace(-r_max, r_max, y_bins + 1), labels=False, include_lowest=True)
    y_sector = 2 * (y_sector / y_bins) - 1

    z_sector = pd.cut(df['z'], bins=np.linspace(0, r_max/10, z_bins + 1), labels=False, include_lowest=True)
    z_sector = 2 * (z_sector / z_bins) - 1

    # Create new DataFrame with the sector information and normalized radius/bearing values
    df_sectors = pd.DataFrame({
        'r': normalized_radius,
        'theta': normalized_theta,
        'r_sector': r_sector,
        'theta_sector': theta_sector,
        'x_sector': x_sector,
        'y_sector': y_sector,
        'z_sector': z_sector
    })

    return df_sectors


def is_smooth(df, threshold):
    """
    Check if the trajectory is smooth.
    Input:
        df: A DataFrame containing the unit vectors of the trajectory.
        threshold: The maximum allowed change in direction angle.
    Output:
        True if the trajectory is smooth, False otherwise.
    """

    # Convert the unit vector columns to a numpy array
    unit_vectors = df[['u_x', 'u_y', 'u_z']].values
    # Calculate the dot product between consecutive unit vectors
    dot_products = cp.einsum('ij, ij->i', unit_vectors[:-1], unit_vectors[1:])
    # Calculate the change in direction angles
    direction_changes = cp.arccos(np.clip(dot_products, -1.0, 1.0)) * 180 / cp.pi
    # Apply modulo operation to keep angles within 360 degrees
    direction_changes = direction_changes % 360

    # Check if any direction change exceeds the threshold
    if cp.max(direction_changes) <= threshold:
        return True
    else:
        return False

def is_flight_path_smooth(df, flight_path_threshold):
    """
    Check if the flight path is smooth.
    Input:
        df: A DataFrame containing the unit vectors of the trajectory.
        flight_path_threshold: The maximum allowed change in flight path angle.
    Output:
        True if the flight path is smooth, False otherwise.
    """

    # Convert the unit vector columns to a CuPy array
    unit_vectors = cp.asarray(df[['u_x', 'u_y', 'u_z']].values)

    # Calculate the change in flight path angles (z-axis)
    flight_path_angles = cp.arcsin(unit_vectors[:, 2]) # arcsin(z) for each unit vector
    flight_path_changes = cp.diff(cp.unwrap(flight_path_angles)) * 180 / cp.pi

    # Check if any flight path change exceeds the threshold
    if cp.max(cp.abs(flight_path_changes)) <= flight_path_threshold:
        return True
    else:
        return False

def calculate_bearing(x, y):
    # Calculate the bearing angle
    bearing_rad = np.arctan2(x, y)

    # Convert the bearing angle from radians to degrees
    bearing_deg = np.degrees(bearing_rad)

    # Adjust the bearing angle to be in the range [0, 360)
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg
