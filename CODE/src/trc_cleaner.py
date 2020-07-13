import pandas as pd
import os
import datetime
from utils import paths
import pickle

def clean(input_path, output_path, pkl_path=None):
    trc_df = pd.read_csv(input_path, index_col=0)
    cols = trc_df.columns
    clustered_dfs = dict()
    cluster_i = 0
    prev_date = None
    for i, (date, df) in enumerate(trc_df.groupby(['Date'])):
        df_reset = df.reset_index(drop=True)
        if prev_date is None:
            prev_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            clustered_dfs[cluster_i] = [prev_date, df_reset]
        else:
            curr_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            date_diff = curr_date - prev_date
            prev_date = curr_date
            if date_diff > datetime.timedelta(days=30):
                cluster_i += 1
                clustered_dfs[cluster_i] = [curr_date, df_reset]
            else:
                last_index = clustered_dfs[cluster_i][1].index[-1]
                df.index = range(last_index + 1, len(df) + last_index + 1)
                clustered_dfs[cluster_i][1] = pd.concat(
                    [clustered_dfs[cluster_i][1], df])

    normalised_dfs = []
    max_first_met = float("-inf")
    min_last_met = float("inf")

    for date, df in clustered_dfs.values():
        date_str = str(date)[:10]

        df['METRAGE'] = df['METRAGE'] * 1000
        df['METRAGE'] = df['METRAGE'].apply(round)
        df['Date'] = pd.Series(list([date_str for __ in range(len(df))]))
        df = df.sort_values(["METRAGE", 'Date'], axis=0)\
            .drop_duplicates(subset=['METRAGE'])

        # if len(df) > 4600:
        # if len(df) > 17000:
        # if len(df) > 2550:
        if len(df) > 135000:
            normalised_dfs.append((date, df))
            metrages = df['METRAGE'].tolist()
            first_metrage, last_metrage = metrages[0], metrages[-1]
            max_first_met = max(first_metrage, max_first_met)
            min_last_met = min(last_metrage, min_last_met)

    aligned_dfs = []
    for date, df in normalised_dfs:
        aligned_df = df[~((df['METRAGE'] < max_first_met) |
                          (df['METRAGE'] > min_last_met))]
        aligned_dfs.append([date, aligned_df])

    filled_dfs = []
    for date, met in aligned_dfs:
        template_df_data = dict()
        template_df_data['METRAGE'] = \
            list(range(max_first_met, min_last_met + 1))
        date = met['Date'].tolist()[0]
        template_df_data['Date'] = list(
            [date for __ in range(len(template_df_data['METRAGE']))])

        template_df = pd.DataFrame.from_dict(template_df_data)
        df_with_nan = pd.concat([met, template_df], keys=('METRAGE'),
                                ignore_index=True, sort=False)\
            .drop_duplicates(subset=['METRAGE'])\
            .sort_values("METRAGE").reset_index(drop=True)
        filled_dfs.append(df_with_nan)
        num_nan = df_with_nan['TOP L'].isna().sum()
        assert num_nan == len(template_df) - len(met)

    merged_df = pd.concat([df for df in filled_dfs], ignore_index=True)\
        .sort_values(["METRAGE", "Date"]).reset_index(drop=True)
    merged_df.to_csv(output_path, na_rep="NaN", index=False, columns=cols)

    if pkl_path:
        with open(pkl_path, 'wb') as file:
            pickle.dump(merged_df, file, -1)


if __name__ == "__main__":
    input_path = os.path.join(paths.DATA, "combined_195.csv")
    output_path = os.path.join(paths.DATA, "C195_consolidated.csv")
    pkl_path = os.path.join(paths.DATA, "C195_from_trc_cleaner.pkl")
    clean(input_path, output_path, pkl_path)


