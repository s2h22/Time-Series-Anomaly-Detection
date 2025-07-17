import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


df = pd.read_csv('pump_sensor_kaggle.csv', index_col=0)

df['timestamp'] = pd.to_datetime(df['timestamp'])
for i in range(52): # sensor from 00 to 51
    df['sensor_%.2d' % i] = pd.to_numeric(df['sensor_%.2d' % i], errors='coerce')

df = df.set_index('timestamp')
df = df.reset_index()

df = df.drop_duplicates() # remove duplicates
del df['sensor_15'] # remove column with no data
del df['sensor_50'] # remove column with no data

df = df.fillna(method='ffill')

normal_df = df[df['machine_status'] == 'NORMAL']
abnormal_df = df[df['machine_status'] != 'NORMAL']

one_tenth = int(len(normal_df)/10)
normal_df1 = df.iloc[:one_tenth*7]
normal_df2 = df.iloc[one_tenth*7:one_tenth*8]
# normal_df3 = df.iloc[one_tenth*8:one_tenth*9]
# normal_df4 = df.iloc[one_tenth*9:]

mean_df = normal_df1.mean(numeric_only=True)
std_df = normal_df1.std(numeric_only=True)

def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size - 1, len(dates)):
        cur_date = dates[idx].to_pydatetime()
        in_date = dates[idx - (window_size - 1)].to_pydatetime()

        _in_period = (cur_date - in_date).days * 24 * 60 + (cur_date - in_date).seconds / 60

        if _in_period == (window_size - 1):
            input_idx.append(list(range(idx - window_size + 1, idx + 1)))
    return input_idx

class TagDataset(Dataset):
    def __init__(self, input_size, df, mean_df=None, std_df=None, window_size=1):
        self.input_size = input_size
        self.window_size = window_size
        original_df = df.copy()

        if mean_df is not None and std_df is not None:
            sensor_columns = [item for item in df.columns if 'sensor_' in item]
            df[sensor_columns] = (df[sensor_columns] - mean_df) / std_df

        dates = list(df['timestamp'])
        self.input_ids = make_data_idx(dates, window_size=window_size)

        self.selected_column = [item for item in df.columns if 'sensor_' in item][:input_size]
        self.var_data = torch.tensor(df[self.selected_column].values, dtype=torch.float)

        self.df = original_df.iloc[np.array(self.input_ids)[:, -1]]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        temp_input_ids = self.input_ids[item]
        input_values = self.var_data[temp_input_ids]
        return input_values