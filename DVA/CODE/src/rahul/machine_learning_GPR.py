import pandas as pd

gpr_filename = "/Users/rahul.chowdhury/omscs/DVA/project/dva_project/data/GPR_tmp.pkl"
gpr_data = pd.read_pickle(gpr_filename)
print(gpr_data)
