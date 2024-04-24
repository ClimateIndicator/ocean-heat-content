import pickle

import numpy as np
import pandas as pd

with open('../Output/IGCC_AR6_update_energy_timeseries_1971to2023_2024-04-13.pickle', 'rb') as filename:
    data = pickle.load(filename)

df = pd.DataFrame(data)
df.to_csv('../Output/IGCC_AR6_update_energy_timeseries_1971to2023_2024-04-13.csv', index=False)
