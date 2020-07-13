## preprocessing functions to apply to TRC data
import pandas as pd

def preprocess_x(df):
    measures = ['AC LN', 'AC VT', 'CON F', 'GAUGE',
       'GR HT', 'OV HT', 'SUPER', 'TOP L', 'TOP R', 'TR ST', 'TW 10', 'TW 3',
       'VER R', 'VOLT', 'culvert', 'Centre',
       'Centre.1', 'Centre.2', 'Centre.3', 'Centre.4', 'Centre.5','Left*', 'Left*.1',
       'Left*.2', 'Left*.3', 'Left*.4', 'Left*.5','Mudspot',
       'PVC Value', 'PVC Value.1', 'PVC Value.2', 'Right', 'Right*',
       'Right*.1', 'Right*.2', 'Right*.3', 'Right*.4', 'Right.1', 'Right.2',
       'Right.3', 'Sleeper_type','Volume (cubic m)', 'Volume (cubic m).1', 'Volume (cubic m).2']

    # measures = ['OV HT', 'SUPER', 'VOLT', 'CON F', 'GAUGE', 'GR HT', 'TR ST', 'VER L',
    #      'TW 10', 'AC LN', 'TOP L', 'TOP R', 'TW 3', 'AC VT']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    ans = df.fillna(0)
    return ans[measures]

def smooth_preds(x,thresh):
    if x > thresh:
        ans = 1
    else:
        ans =0
    return ans
