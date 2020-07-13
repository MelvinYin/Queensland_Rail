from collections import defaultdict
import math
import os
import numpy as np
import pandas as pd
import pickle

from utils import paths

STDEV_LENGTH = 5

def convert(input_pkl):
    with open(input_pkl, 'rb') as file:
        merged_df = pickle.load(file)
    new_dfs = []

    for date_i, (date, df) in enumerate(merged_df.groupby("Date")):
        output = defaultdict(list)
        top_l, top_r, tw_3 = df['TOP L'].values, \
                             df['TOP R'].values, \
                             df['TW 3'].values
        skip_mask = df['TOP L'] != "NaN"
        for i in range(STDEV_LENGTH // 2):
            output['TOP_L_stdev'].append(0)
            output['TOP_R_stdev'].append(0)
            output['TW_3_stdev'].append(0)
        for i in range(STDEV_LENGTH // 2,
                       len(top_l) - ((STDEV_LENGTH // 2) + 1)):
            start_i = i - STDEV_LENGTH // 2
            stop_i = i + ((STDEV_LENGTH // 2) + 1)
            mask = skip_mask[start_i:stop_i]
            if not any(mask):
                err_msg = f"Warning: For date {date}, lines {i} are " \
                          f"empty. This should mean that there are missing " \
                          f"lines of at least {STDEV_LENGTH} in metrage " \
                          f"length in original TRC data."
                print(err_msg)
                output['TOP_L_stdev'].append(0)
                output['TOP_R_stdev'].append(0)
                output['TW_3_stdev'].append(0)
                continue
            top_l_stdev = np.std(top_l[start_i:stop_i][mask.values])
            top_r_stdev = np.std(top_r[start_i:stop_i][mask.values])
            tw_3_stdev = np.std(tw_3[start_i:stop_i][mask.values])
            output['TOP_L_stdev'].append(top_l_stdev)
            output['TOP_R_stdev'].append(top_r_stdev)
            output['TW_3_stdev'].append(tw_3_stdev)
        for i in range(STDEV_LENGTH // 2):
            output['TOP_L_stdev'][i] = output['TOP_L_stdev'][STDEV_LENGTH // 2]
            output['TOP_R_stdev'][i] = output['TOP_R_stdev'][STDEV_LENGTH // 2]
            output['TW_3_stdev'][i] = output['TW_3_stdev'][STDEV_LENGTH // 2]

        final_valid_i = len(output['TOP_L_stdev']) - 1
        for i in range(len(top_l) - ((STDEV_LENGTH // 2) + 1), len(top_l)):
            output['TOP_L_stdev'].append(output['TOP_L_stdev'][final_valid_i])
            output['TOP_R_stdev'].append(output['TOP_R_stdev'][final_valid_i])
            output['TW_3_stdev'].append(output['TW_3_stdev'][final_valid_i])

        assert len(df) == len(output['TOP_L_stdev'])
        assert len(df) == len(output['TOP_R_stdev'])
        assert len(df) == len(output['TW_3_stdev'])
        df['TOP_L_stdev'] = output['TOP_L_stdev']
        df['TOP_R_stdev'] = output['TOP_R_stdev']
        df['TW_3_stdev'] = output['TW_3_stdev']
        new_dfs.append(df)

    merged_df = pd.concat([df for df in new_dfs],
                          ignore_index=True).sort_values(["METRAGE", "Date"])\
        .reset_index(drop=True)
    return merged_df

def align(merged_df, output_path, pkl_path):
    prev_vals = None
    for date_i, (date, df) in enumerate(merged_df.groupby("Date")):
        super_ = df['SUPER'].values
        super_[np.isnan(super_)] = 0
        top_l = df['TOP_L_stdev'].values
        top_r = df['TOP_R_stdev'].values
        tw_3 = df['TW_3_stdev'].values
        if date_i == 0:
            prev_vals = [top_l, top_r, tw_3, super_]
            continue
        prev_l = prev_vals[0]
        prev_r = prev_vals[1]
        prev_3 = prev_vals[2]
        prev_s = prev_vals[3]

        diff_across_60 = []
        for j in range(-100, 100):
            if j < 0:
                assert len(top_l[-j:]) == len(prev_l[:j])
                total_diff = np.sum(np.abs(super_[-j:] - prev_s[:j]))
                assert not math.isnan(total_diff)
                assert total_diff >= 0
                mean_diff = total_diff / len(super_[-j:])
                diff_across_60.append(mean_diff)
            elif j > 0:
                assert len(top_l[:-j]) == len(prev_l[j:])
                total_diff = abs(np.sum(np.abs(super_[:-j] - prev_s[j:])))
                assert not math.isnan(total_diff)
                assert total_diff >= 0
                mean_diff = total_diff / len(top_l[:-j])
                diff_across_60.append(mean_diff)
            else:
                assert j == 0
                assert len(top_l) == len(prev_l)
                total_diff = abs(np.sum(np.abs(super_ - prev_s)))
                assert not math.isnan(total_diff)
                assert total_diff >= 0
                mean_diff = total_diff / len(super_)
                diff_across_60.append(mean_diff)
        to_shift = np.argmax(diff_across_60) - 100

        if to_shift > 0:
            new_l = np.append(top_l[-to_shift:], prev_l[to_shift:])
            new_r = np.append(top_r[-to_shift:], prev_r[to_shift:])
            new_3 = np.append(tw_3[-to_shift:], prev_3[to_shift:])
            new_s = np.append(super_[-to_shift:], prev_s[to_shift:])
        elif to_shift < 0:
            new_l = np.append(prev_l[:to_shift], top_l[:-to_shift])
            new_r = np.append(prev_r[:to_shift], top_r[:-to_shift])
            new_3 = np.append(prev_3[:to_shift], tw_3[:-to_shift])
            new_s = np.append(super_[:to_shift], prev_s[:-to_shift])
        else:
            new_l = top_l
            new_r = top_r
            new_3 = tw_3
            new_s = super_

        merged_df.loc[df['TOP_L_stdev'].index, 'TOP_L_stdev'] = new_l
        merged_df.loc[df['TOP_R_stdev'].index, 'TOP_R_stdev'] = new_r
        merged_df.loc[df['TW_3_stdev'].index, 'TW_3_stdev'] = new_3
        merged_df.loc[df['SUPER'].index, 'SUPER'] = new_s
        prev_vals = [new_l, new_r, new_3, new_s]

    merged_df.to_csv(output_path, na_rep="NaN", index=False)
    if pkl_path:
        with open(pkl_path, 'wb') as file:
            pickle.dump(merged_df, file, -1)

def convert_no_align(input_pkl, output_path, pkl_path):
    with open(input_pkl, 'rb') as file:
        merged_df = pickle.load(file)
    new_dfs = []
    for date, df in merged_df.groupby("Date"):
        output = defaultdict(list)
        metrage, top_l, top_r, tw_3 = df['METRAGE'].values, \
                                      df['TOP L'].values, \
                                      df['TOP R'].values, \
                                      df['TW 3'].values
        skip_mask = df['TOP L'] != "NaN"
        for i in range(STDEV_LENGTH//2):
            output['METRAGE'].append(metrage[i])
            output['TOP_L_stdev'].append(0)
            output['TOP_R_stdev'].append(0)
            output['TW_3_stdev'].append(0)
        for i in range(STDEV_LENGTH // 2, len(metrage)-((STDEV_LENGTH // 2)+1)):
            start_i = i - STDEV_LENGTH // 2
            stop_i = i + ((STDEV_LENGTH // 2) + 1)
            mask = skip_mask[start_i:stop_i]
            if not any(mask):
                err_msg = f"Warning: For date {date}, lines {i} are " \
                          f"empty. This should mean that there are missing " \
                          f"lines of at least {STDEV_LENGTH} in metrage " \
                          f"length in original TRC data."
                print(err_msg)
                output['METRAGE'].append(metrage[i])
                output['TOP_L_stdev'].append(0)
                output['TOP_R_stdev'].append(0)
                output['TW_3_stdev'].append(0)
                continue
            top_l_stdev = np.std(top_l[start_i:stop_i][mask])
            top_r_stdev = np.std(top_r[start_i:stop_i][mask])
            tw_3_stdev = np.std(tw_3[start_i:stop_i][mask])
            output['METRAGE'].append(metrage[i])
            output['TOP_L_stdev'].append(top_l_stdev)
            output['TOP_R_stdev'].append(top_r_stdev)
            output['TW_3_stdev'].append(tw_3_stdev)
        for i in range(STDEV_LENGTH // 2):
            output['TOP_L_stdev'][i] = output['TOP_L_stdev'][STDEV_LENGTH // 2]
            output['TOP_R_stdev'][i] = output['TOP_R_stdev'][STDEV_LENGTH // 2]
            output['TW_3_stdev'][i] = output['TW_3_stdev'][STDEV_LENGTH // 2]

        final_valid_i = len(output['TOP_L_stdev']) - 1
        for i in range(len(metrage) - ((STDEV_LENGTH // 2) + 1), len(metrage)):
            output['METRAGE'].append(metrage[i])
            output['TOP_L_stdev'].append(output['TOP_L_stdev'][final_valid_i])
            output['TOP_R_stdev'].append(output['TOP_R_stdev'][final_valid_i])
            output['TW_3_stdev'].append(output['TW_3_stdev'][final_valid_i])

        assert len(df) == len(output['TOP_L_stdev'])
        assert len(df) == len(output['TOP_R_stdev'])
        assert len(df) == len(output['TW_3_stdev'])
        add_df = pd.DataFrame.from_dict(output).set_index('METRAGE', drop=True)
        df = df.join(add_df, on='METRAGE')
        new_dfs.append(df)

    merged_df = pd.concat([df for df in new_dfs], ignore_index=True)\
        .sort_values(["METRAGE", "Date"])\
        .reset_index(drop=True)

    merged_df.to_csv(output_path, na_rep="NaN", index=False)
    if pkl_path:
        with open(pkl_path, 'wb') as file:
            pickle.dump(merged_df, file, -1)


def convert_user(input_pkl, pkl_path):
    with open(input_pkl, 'rb') as file:
        df = pickle.load(file)
    date = df['Date'].values[0]
    output = defaultdict(list)
    metrage, top_l, top_r, tw_3 = \
        df['METRAGE'].values, df['TOP L'].values, \
        df['TOP R'].values, df['TW 3'].values

    skip_mask = df['TOP L'] != "NaN"
    for i in range(STDEV_LENGTH // 2):
        output['METRAGE'].append(metrage[i])
        output['TOP_L_stdev'].append(0)
        output['TOP_R_stdev'].append(0)
        output['TW_3_stdev'].append(0)
    for i in range(STDEV_LENGTH // 2,
                   len(metrage) - ((STDEV_LENGTH // 2) + 1)):
        start_i = i - STDEV_LENGTH // 2
        stop_i = i + ((STDEV_LENGTH // 2) + 1)
        mask = skip_mask[start_i:stop_i]
        if not any(mask):
            err_msg = f"Warning: For date {date}, lines {i} are " \
                      f"empty. This should mean that there are missing " \
                      f"lines of at least {STDEV_LENGTH} in metrage " \
                      f"length in original TRC data."
            print(err_msg)
            output['METRAGE'].append(metrage[i])
            output['TOP_L_stdev'].append(0)
            output['TOP_R_stdev'].append(0)
            output['TW_3_stdev'].append(0)
            continue
        top_l_stdev = np.std(top_l[start_i:stop_i][mask])
        top_r_stdev = np.std(top_r[start_i:stop_i][mask])
        tw_3_stdev = np.std(tw_3[start_i:stop_i][mask])
        output['METRAGE'].append(metrage[i])
        output['TOP_L_stdev'].append(top_l_stdev)
        output['TOP_R_stdev'].append(top_r_stdev)
        output['TW_3_stdev'].append(tw_3_stdev)
    for i in range(STDEV_LENGTH // 2):
        output['TOP_L_stdev'][i] = output['TOP_L_stdev'][STDEV_LENGTH // 2]
        output['TOP_R_stdev'][i] = output['TOP_R_stdev'][STDEV_LENGTH // 2]
        output['TW_3_stdev'][i] = output['TW_3_stdev'][STDEV_LENGTH // 2]

    final_valid_i = len(output['TOP_L_stdev']) - 1
    for i in range(len(metrage) - ((STDEV_LENGTH // 2) + 1), len(metrage)):
        output['METRAGE'].append(metrage[i])
        output['TOP_L_stdev'].append(output['TOP_L_stdev'][final_valid_i])
        output['TOP_R_stdev'].append(output['TOP_R_stdev'][final_valid_i])
        output['TW_3_stdev'].append(output['TW_3_stdev'][final_valid_i])

    assert len(df) == len(output['TOP_L_stdev'])
    assert len(df) == len(output['TOP_R_stdev'])
    assert len(df) == len(output['TW_3_stdev'])

    if pkl_path:
        with open(pkl_path, 'wb') as file:
            pickle.dump(output, file, -1)

if __name__ == "__main__":
    # merged_df = convert(os.path.join(paths.DATA, "C195_from_trc_cleaner.pkl"))
    #
    # with open("tmp.pkl", 'wb') as file:
    #     pickle.dump(merged_df, file, -1)
    # with open("tmp.pkl", 'rb') as file:
    #     merged_df = pickle.load(file)
    #
    # align(merged_df, os.path.join(paths.DATA, "C195_with_stdev.csv"),
    #              os.path.join(paths.DATA, "C195_with_stdev.pkl"))

    from QRDVA.QRvisualisation.bokeh_ui.figures.trc_heatmap import \
        trc_heatmap_data_preparation

    x = trc_heatmap_data_preparation("C195_with_stdev.pkl")
    with open(os.path.join(paths.DATA, "C195_input.pkl"), 'wb') as file:
        pickle.dump(x, file, -1)
