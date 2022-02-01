## script to create dummy data

import pandas as pd

dummy_df = pd.read_csv("Data/dummy_input_data.csv")
dummy_df = dummy_df[:26281]  # 2 years of data
# print(dummy_df)
times = pd.date_range(start="1/1/2015", end="12/31/2017", freq="1H")
dummy_df['Time [s]'] = times
dummy_df['City'] = 'bs'
dummy_df.to_csv("Data/dummy.csv", index=False)
print(dummy_df)
print("finished")
