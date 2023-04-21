import pickle

import numpy as np
import pandas as pd

with open('../Output/IPCC_AR6_update_energy_timeseries_1971to2022_2023-04-17.pickle', 'rb') as filename:
    data = pickle.load(filename)

df = pd.DataFrame(data)
df.to_csv('../Output/IPCC_AR6_update_energy_timeseries_1971to2022_2023-04-17.csv', index=False)
