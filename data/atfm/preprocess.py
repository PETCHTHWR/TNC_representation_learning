import pandas as pd
from .utils import latlon_to_xy, latlonalt_to_xyz


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


def preprocess(df, ref_lat, ref_lon, ref_alt=0, periods=200, max_range_x = 100, max_range_y=100, alt_column='geoaltitude'):
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
    df_new = rectclip(df_new, max_range_x, max_range_y)
    if len(df_new) == 0:
        return None
    df_new = smooth(df_new)
    df_new = interpolate(df_new, periods=periods)
    #df_new = normalize(df_new, max_range)
    return df_new