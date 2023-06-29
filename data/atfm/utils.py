import numpy as np
import ciso8601
# from numba import jit
from geopy import distance
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import pymap3d as pm
from functools import wraps
from time import time


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


def check_eligible(df, min_alt_change, min_FAF_baro, ad_lat, ad_lon, app_sector_rad):
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
    elif df.iloc[0]['geoaltitude'] == df.iloc[-1]['geoaltitude']:
        return False
    elif df['geoaltitude'].max() - df['geoaltitude'].min() < min_alt_change:
        return False
    elif df['geoaltitude'].min() > min_FAF_baro:
        return False
    elif df['geoaltitude'].max() < min_FAF_baro:
        return False
    elif distance.great_circle((ad_lat, ad_lon), (
    df.loc[df['geoaltitude'].idxmin()]['lat'], df.loc[df['geoaltitude'].idxmin()]['lon'])).nm > app_sector_rad:
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
