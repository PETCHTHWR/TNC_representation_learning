import numpy as np
import pandas as pd

data_path = 'D:/KAIST/KAIST_labwork/ADSB_data/'
arr_csv_path = data_path + 'ADSB_arr.csv'
dep_csv_path = data_path + 'ADSB_dep.csv'

arr_df = pd.read_csv(arr_csv_path)
dep_df = pd.read_csv(dep_csv_path)

