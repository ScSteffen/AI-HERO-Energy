raw_data = pd.read_csv(data_file, delimiter=',')
cities = raw_data['City'].unique()
for city in cities:
    city_data=raw_data[['City'] == city]
    self.dataset = torch.Tensor(city_data['Load [MWh]'].to_numpy())
