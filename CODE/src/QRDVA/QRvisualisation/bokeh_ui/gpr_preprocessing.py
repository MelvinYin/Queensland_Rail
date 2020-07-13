"""
Just for internal documentation



selection_column_mapping = dict()
selection_column_mapping[('LRI', 1)] = 'Left*'
selection_column_mapping[('BTI', 1)] = 'Left*.1'
selection_column_mapping[('MLI', 1)] = 'Left*.2'
selection_column_mapping[('SMLI', 1)] = 'Left*.3'
selection_column_mapping[('FDL', 1)] = 'Left*.4'
selection_column_mapping[('CTQI', 1)] = 'Left*.5'

selection_column_mapping[('LRI', 2)] = 'Centre'
selection_column_mapping[('BTI', 2)] = 'Centre.1'
selection_column_mapping[('MLI', 2)] = 'Centre.2'
selection_column_mapping[('SMLI', 2)] = 'Centre.3'
selection_column_mapping[('FDL', 2)] = 'Centre.4'
selection_column_mapping[('CTQI', 2)] = 'Centre.5'

selection_column_mapping[('LRI', 3)] = 'Right'
selection_column_mapping[('BTI', 3)] = 'Right.1'
selection_column_mapping[('MLI', 3)] = 'Right.2'
selection_column_mapping[('SMLI', 3)] = 'Right.3'
selection_column_mapping[('FDL', 3)] = 'Right.4'
selection_column_mapping[('CTQI', 3)] = 'Right*'

if __name__ == "__main__":
    # sub_divs = ['AT', 'BN', 'CD', 'CY', 'FD', 'FG', 'FS', 'GC', 'IC', 'ML',
    #     'NC', 'NS', 'PA', 'PX', 'QF', 'SF', 'SP', 'SX', 'XX', 'YF']
    #
    # metrics = ['LRI', 'BTI', 'MLI', 'SMLI', 'FDL', 'CTQI']
    # with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
    #     df = pickle.load(file)
    show(GPR().figure)
    # with open(os.path.join(paths.DATA, "gpr_tmp5.pkl"), 'rb') as file:
    #     output = pickle.load(file)
    # print(len(output['region']['xs']))
    # converter = HeatmapRegionConverter()
    # subdiv, long, lat = df['Sub-division'].values, df['Dec.Lat'].values, \
    #                     df['Dec.Long'].values
    # output_arr = defaultdict(list)
    # for div, x, y in zip(subdiv, long, lat):
    #     output_arr['xs'].append(x)
    #     output_arr['ys'].append(y)
    #     output_arr['color'].append(converter.convert(div))
    # output['region']['xs'] = np.array(output_arr['xs'])
    # output['region']['ys'] = np.array(output_arr['ys'])
    # output['region']['color'] = np.array(output_arr['color'])
    #
    # with open(os.path.join(paths.DATA, "gpr_tmp5.pkl"), 'wb') as file:
    #     pickle.dump(output, file, -1)
    #
    # with open(os.path.join(paths.DATA, "gpr_tmp5.pkl"), 'rb') as file:
    #     output = pickle.load(file)
    # output['region']['xs'] = output['region']['xs'][::30]
    # output['region']['ys'] = output['region']['ys'][::30]
    # output['region']['color'] = output['region']['color'][::30]
    # with open(os.path.join(paths.DATA, "gpr_tmp5.pkl"), 'wb') as file:
    #     pickle.dump(output, file, -1)

    # with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
    #     df = pickle.load(file)
    # # classification are:
    # # track (region, left, centre, right) => metric => region
    #
    # output_dict = dict()
    # output_dict['region'] = defaultdict(list)
    # output_dict['left'] = dict()
    # output_dict['centre'] = dict()
    # output_dict['right'] = dict()
    # sub_divs = np.unique(df['Sub-division'].values)
    #
    # for metric in metrics:
    #     output_dict['left'][metric] = dict()
    #     output_dict['centre'][metric] = dict()
    #     output_dict['right'][metric] = dict()
    #
    # for div in sub_divs:
    #     selected_df = df[df['Sub-division'] == div]
    #     output_dict['region']['xs'].extend(list(selected_df[
    #         'Dec.Lat'].values))
    #     output_dict['region']['ys'].extend(list(selected_df[
    #         'Dec.Long'].values))
    #     output_dict['region']['color'].extend(list(selected_df[
    #         'Sub-division'].values))
    #
    #     for metric in metrics:
    #         left_key = selection_column_mapping[(metric, 1)]
    #         centre_key = selection_column_mapping[(metric, 2)]
    #         right_key = selection_column_mapping[(metric, 3)]
    #         if div not in output_dict['left'][metric]:
    #             output_dict['left'][metric][div] = defaultdict(list)
    #         if div not in output_dict['right'][metric]:
    #             output_dict['right'][metric][div] = defaultdict(list)
    #         if div not in output_dict['centre'][metric]:
    #             output_dict['centre'][metric][div] = defaultdict(list)
    #         output_dict['left'][metric][div]['xs'].extend(
    #             list(selected_df['Dec.Lat'].values))
    #         output_dict['left'][metric][div]['ys'].extend(
    #             list(selected_df['Dec.Long'].values))
    #         output_dict['left'][metric][div]['color'].extend(
    #             list(selected_df[left_key].values))
    #
    #         output_dict['right'][metric][div]['xs'].extend(
    #             list(selected_df['Dec.Lat'].values))
    #         output_dict['right'][metric][div]['ys'].extend(
    #             list(selected_df['Dec.Long'].values))
    #         output_dict['right'][metric][div]['color'].extend(
    #             list(selected_df[right_key].values))
    #
    #         output_dict['centre'][metric][div]['xs'].extend(
    #             list(selected_df['Dec.Lat'].values))
    #         output_dict['centre'][metric][div]['ys'].extend(
    #             list(selected_df['Dec.Long'].values))
    #         output_dict['centre'][metric][div]['color'].extend(
    #             list(selected_df[centre_key].values))
    # with open("gpr_tmp.pkl", 'rb') as file:
    #     output_dict = pickle.load(file)
    # with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
    #     df = pickle.load(file)
    # classification are:
    # track (region, left, centre, right) => metric => region

    # print(sub_divs)
    # region_converter = HeatmapRegionConverter()
    # for div in sub_divs:
    #     selected_df = df[df['Sub-division'] == div]
    #     for i in range(len(selected_df['Sub-division'].values)):
    #         output_dict['region']['color'][i] = region_converter.convert(div)
    # with open("gpr_tmp2.pkl", 'wb') as file:
    #     pickle.dump(output_dict, file, -1)
    # with open("gpr_tmp2.pkl", 'rb') as file:
    #     output_dict = pickle.load(file)
    #
    # color_convert = None
    # metrics = ['LRI', 'BTI', 'MLI', 'SMLI', 'FDL', 'CTQI']
    # for div in sub_divs:
    #     for metric in metrics:
    #         if color_convert is None:
    #             values = output_dict['left'][metric][div]['color']
    #             min_, max_ = min(values), max(values)
    #             color_convert = HeatmapColorConverter(min_, max_)
    #         for i in range(len(output_dict['left'][metric][div]['color'])):
    #             output_dict['left'][metric][div]['color'][i] = \
    #                 color_convert.convert(
    #                     output_dict['left'][metric][div]['color'][i])
    #             output_dict['right'][metric][div]['color'][
    #                 i] = color_convert.convert(
    #                 output_dict['right'][metric][div]['color'][i])
    #             output_dict['centre'][metric][div]['color'][
    #                 i] = color_convert.convert(
    #                 output_dict['centre'][metric][div]['color'][i])
    #
    # with open("gpr_tmp3.pkl", 'wb') as file:
    #     pickle.dump(output_dict, file, -1)
    # with open(os.path.join(paths.DATA, "gpr_tmp3.pkl"), 'rb') as file:
    #     output_dict = pickle.load(file)
    # new_output_dict = dict()
    # new_output_dict['xs'] = dict()
    # new_output_dict['ys'] = dict()
    # metrics = ['LRI', 'BTI', 'MLI', 'SMLI', 'FDL', 'CTQI']
    # for div in sub_divs:
    #     new_output_dict['xs'][div] = np.array(output_dict['centre']['LRI'][div][
    #         'xs'])
    #     new_output_dict['ys'][div] = np.array(
    #         output_dict['centre']['LRI'][div]['ys'])
    # for direction in ("left", 'right', 'centre'):
    #     if direction not in new_output_dict:
    #         new_output_dict[direction] = dict()
    #     for div in sub_divs:
    #         if div not in new_output_dict[direction]:
    #             new_output_dict[direction][div] = dict()
    #         for metric in metrics:
    #             if metric not in new_output_dict[direction][div]:
    #                 new_output_dict[direction][div][metric] = dict()
    #             new_output_dict[direction][div][metric] = np.array(
    #                 output_dict[direction][metric][div]['color'])
    # new_output_dict['region'] = output_dict['region']
    #
    # with open("gpr_tmp4.pkl", 'wb') as file:
    #     pickle.dump(new_output_dict, file, -1)

"""