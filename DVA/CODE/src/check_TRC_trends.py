import pandas as pd
from matplotlib import pyplot as plt
from ../utils import paths


trc_data = pd.read_csv(paths.TRC_MA)
print(trc_data.head())