import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, Day


class CustomLoadDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, device=None, normalize=True):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon

        # Load Data from csv to Pandas Dataframe
        raw_data = pd.read_csv(data_file, delimiter=',')['Load [MWh]'].to_numpy()
        self.dataset = torch.Tensor(raw_data)

        # Normalize Data to [0,1]
        if normalize is True:
            self.data_min = torch.min(self.dataset)
            self.data_max = torch.max(self.dataset)
            self.dataset = (self.dataset - self.data_min) / (self.data_max - self.data_min)

        self.dataset = self.dataset.to(device)

    def __len__(self):
        return int(self.dataset.shape[0] - self.historic_window - self.forecast_horizon)

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        x = self.dataset[idx:idx + self.historic_window].unsqueeze(dim=1)
        y = self.dataset[idx + self.historic_window: idx + self.historic_window + self.forecast_horizon].unsqueeze(
            dim=1)

        return x, y

    def revert_normalization(self, data):
        return data * (self.data_max - self.data_min) + self.data_min


class RedWarriorDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, city_='os',device=None, normalize=True):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon
        self.normalize=normalize
        self.city = city_
        self.city_population = {'h': 535061.0,
                                'bs': 248023.0,
                                'ol': 167081.0,
                                'os': 164374.0,
                                'wob': 123914.0,
                                'go': 119529.0,
                                'sz': 104548.0,
                                'hi': 101744.0,
                                'del': 77521.0,
                                'lg': 75192.0,
                                'whv': 76316.0,
                                'ce': 69706.0,
                                'hm': 57228.0,
                                'el': 54117.0}

        # Load Data from csv to Pandas Dataframe
        raw_data = pd.read_csv(data_file, delimiter=',')
        raw_data['Time [s]'] = pd.to_datetime(raw_data['Time [s]'])

        raw_data['day_frac'] = (raw_data['Time [s]'] - pd.to_datetime(raw_data['Time [s]'].dt.date)).dt.total_seconds() / (
                24 * 3600)

        raw_data['week_frac'] = (raw_data['Time [s]'].dt.dayofweek + raw_data['frac_day']) / 7

        raw_data['month_frac'] = (raw_data['Time [s]'].dt.day + raw_data['frac_day'] - 1) / raw_data[
            'Time [s]'].dt.days_in_month

        raw_data['year_frac'] = raw_data['Time [s]'].dt.dayofyear / (365 + raw_data['Time [s]'].dt.is_leap_year.astype(float))

        self.cities = raw_data['City'].unique()
        self.city_pop_in_data = {x: self.city_population[x] for x in self.city_population if x in self.cities}

        self.dataset = {}
        for city, population in self.city_pop_in_data:
            city_data = raw_data[raw_data['City'] == city]
            self.dataset[city] = torch.Tensor(city_data['Load [MWh]',
                                                        'day_frac',
                                                        'week_frac',
                                                        'month_frac',
                                                        'year_frac',].values.astype(np.float32)) / population
            # Normalize Data to [0,1]
            if normalize is True:
                self.data_min[city] = torch.min(self.dataset[city])
                self.data_max[city] = torch.max(self.dataset[city])
                self.dataset[city] = (self.dataset[city] - self.data_min[city]) / (
                        self.data_max[city] - self.data_min[city])
            self.dataset[city] = self.dataset.to(device)

    def __len__(self):
        return int(next(iter(self.dataset)).shape[0] - self.historic_window - self.forecast_horizon)

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        x = self.dataset[self.city][idx:idx + self.historic_window]
        y = self.dataset[self.city][idx + self.historic_window: idx + self.historic_window + self.forecast_horizon]

        return x, y

    def revert_normalization(self, data):
        data_ret = {}
        for city, data_ in data:
            # Normalize Data to [0,1]
            data_ret = data_ * (self.data_max[city] - self.data_min[city]) + self.data_min[city]
            return data_ret
        else:
            return data
