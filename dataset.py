import torch
from torch.utils.data import Dataset
import pandas as pd
import statistics
import numpy as np

def remove_outliers2(df):
    window = 7*24
    sign = lambda x: (1, -1)[x<0]
    for k in range(0,len(df['Load [MWh]'])-window,window):
        m = statistics.mean(df['Load [MWh]'][k:k+window])
        s = statistics.stdev(df['Load [MWh]'][k:k+window])
        for j in range(k,k+window,1):
            dfv = df.iloc[j, df.columns.get_loc('Load [MWh]')]
            if (np.abs(dfv-m) > 2*s):
                sig = sign(dfv-m)
                df.iloc[j, df.columns.get_loc('Load [MWh]')] = m + sig*2*s

    return df

class CustomLoadDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, device=None, normalize=True):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon

        # Load Data from csv to Pandas Dataframe
        raw_data = pd.read_csv(data_file, delimiter=',')

        raw_data['Time [s]'] = pd.to_datetime(raw_data['Time [s]'])

        raw_data['day_frac'] = (raw_data['Time [s]'] - pd.to_datetime(
            raw_data['Time [s]'].dt.date)).dt.total_seconds() / (
                                       24 * 3600)

        raw_data['week_frac'] = (raw_data['Time [s]'].dt.dayofweek + raw_data['day_frac']) / 7

        # raw_data['month_frac'] = (raw_data['Time [s]'].dt.day + raw_data['day_frac'] - 1) / raw_data[
        #     'Time [s]'].dt.days_in_month

        raw_data['year_frac'] = raw_data['Time [s]'].dt.dayofyear / (
                365 + raw_data['Time [s]'].dt.is_leap_year.astype(float))

        # Group data by city
        groups = raw_data.groupby('City')
        cities = []
        for city, df in groups:
            cities.append(torch.tensor(df[['Load [MWh]', \
                                           'day_frac', \
                                           'week_frac', \
                                           # 'month_frac',
                                           'year_frac']].to_numpy(), dtype=torch.float))

        # Generate data tensor and metadata
        self.dataset = torch.stack(cities)
        self.city_nr = self.dataset.shape[0]
        self.samples_per_city = self.dataset.shape[1] - self.historic_window - self.forecast_horizon

        # Normalize Data to [0,1]
        if normalize is True:
            self.data_min = torch.min(self.dataset[:,:,0])
            self.data_max = torch.max(self.dataset[:,:,0])
            self.dataset = (self.dataset - self.data_min) / (self.data_max - self.data_min)

        self.dataset = self.dataset.to(device)

    def __len__(self):
        return self.city_nr * self.samples_per_city

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        city_idx = idx // self.samples_per_city
        hour_idx = idx % self.samples_per_city
        x = self.dataset[city_idx, hour_idx:hour_idx+self.historic_window]
        y = self.dataset[city_idx, hour_idx+self.historic_window:
                                   hour_idx+self.historic_window + self.forecast_horizon, 0].unsqueeze(dim=1)

        return x, y

    def revert_normalization(self, data):
        return data[:,0] * (self.data_max - self.data_min) + self.data_min
