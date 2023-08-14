import pandas as pd
from .utils import latlon_to_xy, latlonalt_to_xyz
from scipy.signal import savgol_filter
from scipy import stats
import numpy as np

def baroprojection(df, ref_lat, ref_lon, ref_alt):
    """
    Project (lat, lon, baroaltitude) of pandas dataframe to (x, y, z)
    :param df: pandas dataframe containing columns 'lat', 'lon', 'baroaltitude'
    :param ref_lat: reference latitude
    :param ref_lon: reference longitude
    :param ref_alt: reference altitude
    :return: pandas dataframe
    """
    # Copy df
    new_df = pd.DataFrame(index=df.index)
    new_df[['x', 'y', 'z']] = df[['lat', 'lon', 'baroaltitude']].apply(
        lambda x: latlon_to_xy(x[0], x[1], ref_lat, ref_lon) + (x[2] - ref_alt,), axis=1, result_type='expand')
    return new_df

def geoprojection(df, ref_lat, ref_lon, ref_alt):
    """
    Project (lat, lon, baroaltitude) of pandas dataframe to (x, y, z)
    :param df: pandas dataframe containing columns 'lat', 'lon', 'baroaltitude'
    :param ref_lat: reference latitude
    :param ref_lon: reference longitude
    :param ref_alt: reference altitude
    :return: pandas dataframe
    """
    # Copy df
    new_df = pd.DataFrame(index=df.index)
    new_df[['x', 'y', 'z']] = df[['lat', 'lon', 'geoaltitude']].apply(
        lambda x: latlon_to_xy(x[0], x[1], ref_lat, ref_lon) + (x[2] - ref_alt,), axis=1, result_type='expand')
    return new_df


def smooth(df):
    """
    Smooth the trajectory using median filter
    :param df: pandas dataframe containing columns 'x', 'y', 'z'
    :return: pandas dataframe
    """
    df['x'] = df['x'].rolling(5, center=True, min_periods=1).median()
    df['y'] = df['y'].rolling(5, center=True, min_periods=1).median()
    df['z'] = df['z'].rolling(5, center=True, min_periods=1).median()
    return df

def smooth_savgol(df, window_size=7, order=2):
    """
    Smooth the trajectory using Savitzky-Golay filter
    :param df: pandas DataFrame containing columns 'x', 'y', 'z'
    :param window_size: size of the smoothing window
    :param order: order of the polynomial
    :return: pandas DataFrame
    """
    smoothed_df = df.copy()

    smoothed_df['x'] = savgol_filter(df['x'].values, window_size, order)
    smoothed_df['y'] = savgol_filter(df['y'].values, window_size, order)
    smoothed_df['z'] = savgol_filter(df['z'].values, window_size, order)

    return smoothed_df

def smooth_savgol_vel(df, window_size=7, order=2):
    """
    Smooth the trajectory using Savitzky-Golay filter
    :param df: pandas DataFrame containing columns 'x', 'y', 'z'
    :param window_size: size of the smoothing window
    :param order: order of the polynomial
    :return: pandas DataFrame
    """
    smoothed_df = df.copy()

    smoothed_df['v_x'] = savgol_filter(df['v_x'].values, window_size, order)
    smoothed_df['v_y'] = savgol_filter(df['v_y'].values, window_size, order)
    smoothed_df['v_z'] = savgol_filter(df['v_z'].values, window_size, order)

    return smoothed_df

# @timeit
def interpolate(df, periods=200):
    """
    Resample the trajectory to a given frequency
    :param df: pandas dataframe to be resampled
    :param freq: frequency, default '1s'
    :return: pandas dataframe
    """
    desired_index = pd.date_range(df.index[0], df.index[-1], periods=periods)
    df = df.reindex(df.index.union(desired_index))
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.reindex(desired_index)
    return df

def resample(df, freq='1s'):
    """
    Resample the trajectory to a given frequency
    :param df: pandas dataframe to be resampled
    :param freq: frequency, default '1s'
    :return: pandas dataframe
    """
    df = df.resample(freq).mean()
    return df


def remove_outliers(df, threshold=3):
    df = df.copy()  # To ensure we don't modify the original dataframe
    df.loc[:, 'z_diff'] = df['z'].diff()
    df = df.dropna()
    df.loc[:, 'z_diff_z'] = stats.zscore(df['z_diff'].abs())
    df = df[df['z_diff_z'].abs() <= threshold]
    df = df.drop(columns=['z_diff', 'z_diff_z'])
    return df


def remove_outliers_vel(df, threshold=3):
    df = df.copy()  # To ensure we don't modify the original dataframe
    cols = ['v_x', 'v_y', 'v_z']
    for col in cols:
        diff_col_name = f'{col}_diff'
        diff_z_col_name = f'{col}_diff_z'

        df[diff_col_name] = df[col].diff().abs()
        df[diff_z_col_name] = stats.zscore(df[diff_col_name])

        # Identify outliers
        outliers = df[np.abs(df[diff_z_col_name]) > threshold]

        # Replace outliers with NaN
        for outlier in outliers.index:
            df.loc[outlier, col] = np.nan

        # Interpolate missing values
        df[col].interpolate(method='linear', inplace=True)

        # drop intermediate columns
        df = df.drop(columns=[diff_col_name, diff_z_col_name])

    return df


def clip(df, max_range):
    """
    Clip the trajectory to a given distance from the reference point
    :param df: pandas dataframe containing columns 'x', 'y'
    :param max_distance: maximum distance from the reference point
    :return:
    """
    inside = df['x'] ** 2 + df['y'] ** 2 < (max_range * 1000) ** 2
    # Clip from one row before first point inside until one row after last point inside
    if not inside.any():
        return None
    start = inside[inside].index[0]
    end = inside[inside].index[-1]
    df = df[start:end]
    return df

def rectclip(df, max_range_x, max_range_y):
    """
    Clip the trajectory to a given distance from the reference point
    :param df: pandas dataframe containing columns 'x', 'y'
    :param max_distance: maximum distance from the reference point
    :return:
    """
    inside = (df['x'].abs() < max_range_x * 1000) & (df['y'].abs() < max_range_y * 1000)
    # Clip from one row before first point inside until one row after last point inside
    if not inside.any():
        return None
    start = inside[inside].index[0]
    end = inside[inside].index[-1]
    df = df[start:end]
    return df

def normalize(df, max_range):
    df['x'] = df['x'] / (max_range * 1000)
    df['y'] = df['y'] / (max_range * 1000)
    df['z'] = df['z'] / 12000
    return df


def resample_and_trim_trajectory(df, target_length=1500):
    # Ensure the DataFrame is sorted by the index
    df = df.sort_index()

    # Resample the dataframe to 1 second intervals
    resampled_df = df.resample('1S').interpolate()

    # If the resampled dataframe is longer than the target length, trim it
    if len(resampled_df) > target_length:
        resampled_df = resampled_df.iloc[:target_length]

    return resampled_df

def preprocess(df, ref_lat, ref_lon, ref_alt=0, periods=200, max_range = 100,
               alt_column='geoaltitude', freq='1s', zscore_threshold=3, window_size=7, order=2, unify=True):
    """
    Preprocess the trajectory
    :param df: pandas dataframe containing columns 'lat', 'lon', alt_column
    :param ref_lat: reference latitude
    :param ref_lon: reference longitude
    :param ref_alt: reference altitude, default 0
    :param max_range: maximum distance from the reference point, default 100 km
    :param alt_column: column name for altitude, default 'baroaltitude'
    :return: pandas dataframe
    """
    # Copy df
    if len(df) == 0:
        return None
    df_new = df.copy()

    if alt_column == 'geoaltitude':
        df_new = geoprojection(df_new, ref_lat, ref_lon, ref_alt)
    else:
        df_new = baroprojection(df_new, ref_lat, ref_lon, ref_alt)

    df_new = resample(df_new, freq=freq) # resample to 1s
    df_new = remove_outliers(df_new, threshold=zscore_threshold).interpolate() # remove outliers and interpolate
    df_new = clip(df_new, max_range)
    if len(df_new) == 0:
        return None # remove outliers

    if unify:
        df_new = interpolate(df_new, periods=periods) # interpolate to 200 points

    df_new = smooth_savgol(df_new, window_size=window_size, order=order)

    #df_new = normalize(df_new, max_range_x)
    return df_new