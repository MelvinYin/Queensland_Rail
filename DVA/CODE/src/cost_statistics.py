from utils import paths
import pandas as pd
from matplotlib import pyplot as plt
COST_CSV_FILE = paths.DATA_DERIVED+"/work_orders_with_cost.csv"
data_xls = pd.read_excel(paths.DATA_OCT+"/Work Orders for Corridors C138 and C195 10.10.2019 including costs.xlsx",index_col=None,usecols=["Basic fin. date","TotalPlnndCosts","Total act.costs"])
data_xls.to_csv(COST_CSV_FILE,encoding='utf-8',index=False)

csv_data = pd.read_csv(COST_CSV_FILE)
csv_data.plot(kind="bar")
plt.show()
print(csv_data.head())