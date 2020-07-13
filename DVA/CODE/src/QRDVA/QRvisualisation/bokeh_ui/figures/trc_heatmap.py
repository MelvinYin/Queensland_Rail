import pickle

STDEV_LENGTH = 0.005

import os

from bokeh.plotting import figure
from bokeh.models.widgets import HTMLTemplateFormatter
heat_width = 0.01
from bokeh.layouts import row, column
from bokeh.models import RangeTool
from utils import paths

try:
    from ...bokeh_ui.figures.components import heatmap_c, boxes_c
except ValueError:
    from QRDVA.QRvisualisation.bokeh_ui.figures.components import heatmap_c, \
     boxes_c

import numpy as np
from bokeh.models.widgets import Select
from collections import defaultdict
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn


def extract_wo_key_date_met(input_trc_pkl="C195_with_stdev.pkl"):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)
    wo_key_date_met = defaultdict(dict)
    for _i, pd_series in outputs.iterrows():
        wo = pd_series['Work_orders']
        if isinstance(wo, str) and wo != "[]":
            METRAGE = pd_series['METRAGE']
            date = pd_series['Date']
            if "," not in wo:
                wo_key = wo[1:-1]
                if date not in wo_key_date_met[wo_key]:
                    wo_key_date_met[wo_key][date] = []
                wo_key_date_met[wo_key][date].append(METRAGE)

            else:
                wo_key_list = wo[1:-1]
                wo_keys = wo_key_list.split(",")
                for key in wo_keys:
                    key = key.strip()
                    if date not in wo_key_date_met[key]:
                        wo_key_date_met[key][date] = []
                    wo_key_date_met[key][date].append(METRAGE)
    return wo_key_date_met


def extract_wo_key_type(input_trc_pkl="C195_with_stdev.pkl"):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)
    wo_key_type = dict()
    for _i, pd_series in outputs.iterrows():
        wo = pd_series['Work_orders']
        if isinstance(wo, str) and wo != "[]":
            if "," not in wo:
                wo_key = wo[1:-1]

                wo_type = pd_series['Work_order_type'][1:-1]
                if wo_key in wo_key_type:
                    assert wo_key_type[wo_key] == wo_type
                else:
                    wo_key_type[wo_key] = wo_type
            else:
                wo_key_list = wo[1:-1]
                wo_keys = wo_key_list.split(",")
                wo_types = pd_series['Work_order_type'][1:-1]
                assert "," in wo_types
                wo_types = wo_types.split(",")
                for wo_key, wo_type in zip(wo_keys, wo_types):
                    wo_key = wo_key.strip()
                    wo_type = wo_type.strip()
                    if wo_key in wo_key_type:
                        assert wo_key_type[wo_key] == wo_type
                    else:
                        wo_key_type[wo_key] = wo_type
    return wo_key_type


def get_wo_type_color():
    wo_type_color = dict()
    wo_type_color["'Ballast Undercutting'"] = "#00ff00"
    wo_type_color["'Formation repairs'"] = "#013220"
    wo_type_color["'Maintenance Ballasting'"] = "#add8e6"
    wo_type_color["'Mechanised Ballasting'"] = "#0000FF"
    wo_type_color["'Mechanised Resleepering'"] = "#0000A0"
    wo_type_color["'Mechanised Resurfacing'"] = "#800080"
    wo_type_color["'Top & Line Spot Resurfacing'"] = "#FF0000"
    return wo_type_color


def get_dates(input_trc_pkl="C195_with_stdev.pkl"):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)
    return list(sorted(set(outputs['Date'].values)))

def create_wo_input(input_trc_pkl="C195_with_stdev.pkl", output="wo_input.pkl"):
    dates = get_dates(input_trc_pkl)
    wo_key_date_met = extract_wo_key_date_met(input_trc_pkl)
    wo_key_type = extract_wo_key_type(input_trc_pkl)
    wo_type_color = get_wo_type_color()

    with open('tmp.pkl', 'wb') as file:
        pickle.dump((wo_key_date_met, wo_key_type), file, -1)

    with open("tmp.pkl", 'rb') as file:
        wo_key_date_met, wo_key_type = pickle.load(file)

    wo_input_raw = dict()
    for wo_key, date_met in wo_key_date_met.items():
        assert wo_key in wo_key_type
        dates_per_wo = list(date_met.keys())
        assert len(dates_per_wo) == 1
        metrages = date_met[dates_per_wo[0]]
        assert wo_key_type[wo_key] in wo_type_color
        wo_input_raw[wo_key] = (min(metrages), max(metrages),
                           wo_type_color[wo_key_type[wo_key]], dates_per_wo[0])

    dates_i_map = dict()
    for i, date in enumerate(dates):
        dates_i_map[date] = len(dates) - i - 1

    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)
    metrages = set(outputs['METRAGE'].values)
    min_metrage, max_metrage = min(metrages), max(metrages)
    if round(min_metrage, -3) > min_metrage:
        min_metrage_floor = round(min_metrage, -3) - 1000
    else:
        min_metrage_floor = round(min_metrage, -3)

    metrage_map = dict()
    i = 0
    curr_val = min_metrage_floor
    while True:
        metrage_map[i] = curr_val
        if curr_val > max_metrage:
            break
        curr_val += 1000
        i += 1

    wo_input = [defaultdict(list) for __ in range(len(metrage_map))]
    for split_i, met_range_start in metrage_map.items():
        met_range_end = met_range_start + 1000
        key_to_delete = []
        key_to_add = []
        for wo_key, values in wo_input_raw.items():
            met_start, met_end, wo_type, date = values
            assert met_end >= met_start
            if met_start == met_end:
                key_to_delete.append(wo_key)
                continue
            if met_start >= met_range_start and met_start < met_range_end:
                y1 = dates_i_map[date]
                y2 = dates_i_map[date] + len(dates) + 1
                y3 = dates_i_map[date] + 2 * len(dates) + 2
                color = wo_type
                if met_end <= met_range_end:
                    x = (met_start + met_end) // 2
                    width = met_end - met_start
                    key_to_delete.append(wo_key)
                else:
                    x = (met_start + met_range_end) // 2
                    width = met_range_end - met_start
                    key_to_add.append(
                        [wo_key, met_range_end, met_end, wo_type, date])

                wo_input[split_i]['xs'].extend([x, x, x])
                wo_input[split_i]['ys'].extend([y1, y2, y3])
                wo_input[split_i]['color'].extend([color, color, color])
                wo_input[split_i]['width'].extend([width, width, width])

        for key in key_to_delete:
            del wo_input_raw[key]
        for term in key_to_add:
            wo_input_raw[term[0]] = term[1:]
    assert not wo_input_raw
    with open(os.path.join(paths.DATA, output), 'wb') as file:
        pickle.dump(wo_input, file, -1)

def wo_type_map_data_preparation():
    wo_type_map = get_wo_type_color()
    wo_map_input = defaultdict(list)
    colours = ['Green', 'Dark Green', 'Light Blue', 'Blue', 'Dark Blue', 'Purple',
               'Red']
    for i, (wo_type, color_code) in enumerate(wo_type_map.items()):
        wo_map_input['wo_type'].append(wo_type[1:-1])
        wo_map_input['color'].append((color_code, colours[i]))
    with open(os.path.join(paths.DATA, "wo_type_map.pkl"), 'wb') as file:
        pickle.dump(wo_map_input, file, -1)


def trc_heatmap_data_preparation(input_trc_pkl="C195_with_stdev.pkl"):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)
    metrage_full = outputs['METRAGE'].drop_duplicates().values

    hm_color = heatmap_c.HeatmapColorConverter(0, 10)

    offset_left = int(round(min(metrage_full), -3) - min(metrage_full))
    if offset_left < 0:
        offset_left += 1000

    num_metrage_splits = (max(metrage_full) - min(metrage_full)) // 1000 - 1
    num_dates = len(np.unique(outputs['Date'].values))

    output_by_property = list(
        [defaultdict(list) for __ in range(num_metrage_splits)])

    dates = []
    for split_i in range(num_metrage_splits):
        start_metrage = offset_left + min(metrage_full) + split_i * 1000
        end_metrage = offset_left + min(metrage_full) + (split_i + 1) * 1000
        selected_df_all_dates = outputs[
            (outputs['METRAGE'] >= start_metrage) & (
                        outputs['METRAGE'] < end_metrage)]
        for date_i, (date, selected_df) in enumerate(
            selected_df_all_dates.groupby("Date")):
            if len(dates) != num_dates:
                dates.append(date)

            maskl = selected_df['TOP_L_stdev'].values > 2
            maskr = selected_df['TOP_R_stdev'].values > 2
            mask3 = selected_df['TW_3_stdev'].values > 2

            top_l_data = list(map(hm_color.convert, selected_df[
                'TOP_L_stdev'].values[maskl]))
            top_r_data = list(map(hm_color.convert, selected_df['TOP_R_stdev'].values[
                maskr]))
            tw_3_data = list(map(hm_color.convert, selected_df['TW_3_stdev'].values[
                mask3]))
            # print(len(selected_df['TW_3_stdev'].values[mask]))
            # print(len(selected_df['TW_3_stdev'].values))
            # print("")
            metrage_local_l = np.array(range(start_metrage, end_metrage))[maskl]
            metrage_local_r = np.array(range(start_metrage, end_metrage))[maskr]
            metrage_local_3 = np.array(range(start_metrage, end_metrage))[mask3]

            y1 = np.array([date_i for __ in range(1000)])[maskl]
            y2 = np.array([date_i + num_dates + 1 for __ in range(1000)])[maskr]
            y3 = np.array([date_i + 2 * num_dates + 2 for __ in range(1000)])[
                mask3]

            assert len(metrage_local_l) == len(y1)
            assert len(metrage_local_r) == len(y2)
            assert len(metrage_local_3) == len(y3)
            assert len(metrage_local_l) == len(top_l_data)
            assert len(metrage_local_r) == len(top_r_data)
            assert len(metrage_local_3) == len(tw_3_data)

            output_by_property[split_i]['xs'].extend(list(metrage_local_l))
            output_by_property[split_i]['ys'].extend(list(y1))
            output_by_property[split_i]['color'].extend(list(top_l_data))

            output_by_property[split_i]['xs'].extend(list(metrage_local_r))
            output_by_property[split_i]['ys'].extend(list(y2))
            output_by_property[split_i]['color'].extend(list(top_r_data))

            output_by_property[split_i]['xs'].extend(list(metrage_local_3))
            output_by_property[split_i]['ys'].extend(list(y3))
            output_by_property[split_i]['color'].extend(list(tw_3_data))

    output_for_background = list(
        [dict() for __ in range(num_metrage_splits)])
    for split_i, values_dict in enumerate(output_by_property):
        xs_start = offset_left + min(metrage_full) + split_i * 1000
        xs = [xs_start + 500 for __ in range(3)]
        ys = [len(dates) / 2 - 0.5, len(dates) * 3 / 2 + 0.5,
              len(dates) * 5 / 2 + 1.5]
        color = ['#FFA500', '#FFA500', '#FFA500']
        output_for_background[split_i]['xs'] = xs
        output_for_background[split_i]['ys'] = ys
        output_for_background[split_i]['color'] = color

    return output_by_property, dates, output_for_background

class DropDownComponent_mine:
    def __init__(self, specs, callback=None):
        self.specs = specs
        self.figure = self._set_dropdown(callback)

    def _set_dropdown(self, callback):
        dropdown = Select(title=self.specs["title"],
                          value=self.specs['menu'][0][0],
                          options=self.specs['menu'], width=self.specs['width'],
                          height=self.specs['height'])
        if callback:
            dropdown.js_on_change('value', callback)
        return dropdown


class TRCHeatMap:
    def __init__(self, heatmap_dates, title, wo_data, wo_type_color_map):
        self.title = title
        self.heatmap_data = heatmap_dates[0]
        self.starting_offset = self.heatmap_data[0]['xs'][0]
        self.wo_data = wo_data
        self.dates = heatmap_dates[1]
        self.background_data = heatmap_dates[2]
        self.wo_type_color_map = wo_type_color_map
        self.cds = self._build_cds()
        self.background_cds = self._build_background_cds()
        self.wo_cds = self._build_wo_cds()
        self.zoomed_heatmap = self._build_zoomed_heatmap()
        self.rangetool = self._build_rangetool()
        self.full_heatmap = self._build_full_heatmap()
        self.dropdown = self._build_dropdown()
        self.wo_table = self._build_wo_type_color_table()
        self.yaxis_labels_1 = self._build_yaxis_labels_1()
        self.yaxis_labels_2 = self._build_yaxis_labels_2()
        self.figure = self._build_figure()

    def _build_wo_cds(self):
        cds = ColumnDataSource(data=self.wo_data[1])
        return cds

    def _build_background_cds(self):
        cds = ColumnDataSource(data=self.background_data[0])
        return cds

    def _build_yaxis_labels_1(self):
        tb0_specs = dict(title="", width=60, height=35)
        tb0 = boxes_c.TextBoxComponent(tb0_specs)
        tb1_specs = dict(title="TOP_L", width=60, height=65)
        tb1 = boxes_c.TextBoxComponent(tb1_specs)
        tb2_specs = dict(title="TOP_R", width=60, height=65)
        tb2 = boxes_c.TextBoxComponent(tb2_specs)
        tb3_specs = dict(title="TW_3", width=60, height=65)
        tb3 = boxes_c.TextBoxComponent(tb3_specs)
        label_column = column([tb0.figure, tb1.figure, tb2.figure, tb3.figure],
                              width=80, height=200)
        return label_column

    def _build_yaxis_labels_2(self):
        tb0_specs = dict(title="", width=60, height=75)
        tb0 = boxes_c.TextBoxComponent(tb0_specs)
        tb1_specs = dict(title="TOP_L", width=60, height=170)
        tb1 = boxes_c.TextBoxComponent(tb1_specs)
        tb2_specs = dict(title="TOP_R", width=60, height=170)
        tb2 = boxes_c.TextBoxComponent(tb2_specs)
        tb3_specs = dict(title="TW_3", width=60, height=180)
        tb3 = boxes_c.TextBoxComponent(tb3_specs)
        label_column = column([tb0.figure, tb1.figure, tb2.figure, tb3.figure],
                              width=80, height=200)
        return label_column

    def _build_dropdown(self):
        dropdown_specs = dict()
        dropdown_specs["title"] = "Metrage Range Selection"
        dropdown_specs['menu'] = [(str(i), f"{i * 1000 + self.starting_offset}-"
                                           f"{(i + 1) * 1000 + self.starting_offset}") for i in
                                  range(len(self.heatmap_data))]
        dropdown_specs['width'] = 250
        dropdown_specs['height'] = 100
        rangetool_range = self.rangetool.x_range.end - \
                          self.rangetool.x_range.start

        from bokeh.models.callbacks import CustomJS

        callback = CustomJS(args=dict(cds=self.cds,
                                      background_cds=self.background_cds,
                                      background_data=self.background_data,
                                      hm_data=self.heatmap_data,
                                      wo_data=self.wo_data,
                                      wo_cds=self.wo_cds,
                                      f_hm=self.full_heatmap,
                                      rangetool=self.rangetool,
                                      rangetool_range=rangetool_range,
                                      offset=self.starting_offset//1000),
                            code="""
        var f = cb_obj.value;
        cds.data = hm_data[f];
        background_cds.data = background_data[f];
        wo_cds.data = wo_data[Number(f)+1];
        f_hm.x_range.start = (Number(f)+offset)*1000;
        f_hm.x_range.end = (Number(f)+offset+1)*1000;
        rangetool.x_range.start = (Number(f)+offset) * 1000;
        rangetool.x_range.end = (Number(f)+offset) * 1000 + rangetool_range;
        """)
        # cds.change.emit();
        downdown = DropDownComponent_mine(dropdown_specs, callback)
        return downdown

    def _build_cds(self):
        cds = ColumnDataSource(data=self.heatmap_data[0])
        return cds

    def _build_rangetool(self):
        range_tool = RangeTool(x_range=self.zoomed_heatmap.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2
        return range_tool

    def _build_zoomed_heatmap(self):
        zoomed_hm = figure(title=f"{self.title} Zoomed", plot_height=300,
                           plot_width=300, x_axis_location="below",
                           background_fill_color="#efefef",
                           x_range=(self.starting_offset,
                                    self.starting_offset+50),
                           y_range=[len(self.dates)*3+2, -1],
                           tools='save')

        zoomed_hm.yaxis.major_label_text_font_size = '4pt'
        zoomed_hm.xaxis.ticker = []
        ticker_range = []
        ticker_range.extend(list(range(len(self.dates))))
        ticker_range.extend(
            list(range(len(self.dates) + 1, 2 * len(self.dates) + 1)))
        ticker_range.extend(
            list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)))
        zoomed_hm.yaxis.ticker = ticker_range
        override_dict = dict()
        for i, date in zip(list(range(len(self.dates))), self.dates):
            override_dict[i] = date
        for i, date in zip(
                list(range(len(self.dates) + 1, 2 * len(self.dates) + 1)),
                self.dates):
            override_dict[i] = date

        for i, date in zip(
                list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)),
                self.dates):
            override_dict[i] = date

        zoomed_hm.yaxis.major_label_overrides = override_dict
        # zoomed_hm.rect([min(self.cds.data['xs']) + 500 for __ in range(3)],
        #                [len(self.dates) / 2 - 0.5,
        #                 len(self.dates) * 3 / 2 + 0.5,
        #                 len(self.dates) * 5 / 2 + 1.5],
        #                color=['#FFA500', '#FFA500', '#FFA500'], width=1000,
        #                height=len(self.dates))
        zoomed_hm.rect('xs', 'ys', color='color', source=self.background_cds, width=1000,
                       height=len(self.dates))
        zoomed_hm.rect('xs', 'ys', color='color', source=self.cds, width=1,
                       height=1)
        zoomed_hm.rect('xs', 'ys', color='color', source=self.wo_cds,
                       width='width',
                       height=1, fill_alpha=0, line_alpha=1)
        return zoomed_hm

    def _build_wo_type_color_table(self):
        template = """
        <div style="background:<%= value[0] %>; 
            color: white"> 
        <%= value[1]%></div>
        """
        formatter = HTMLTemplateFormatter(template=template)
        cds = ColumnDataSource(self.wo_type_color_map)
        columns = [TableColumn(field="wo_type", title="Work Order Type",
                               width=150),
                   TableColumn(field="color", title="Colour",
                               formatter=formatter, width=80)]
        table = DataTable(source=cds, columns=columns, height=600, width=230,
                      fit_columns=True, sortable=False, reorderable=False,
                      selectable=False, index_position=None)
        return table


    def _build_full_heatmap(self):
        full_hm = figure(title=f"{self.title} Full", height=600, width=800,
                         x_range=[self.starting_offset,
                                  self.starting_offset+1000],
                         y_range=self.zoomed_heatmap.y_range,
                         tools='save')
        ticker_range = []
        ticker_range.extend(list(range(len(self.dates))))
        ticker_range.extend(
            list(range(len(self.dates) + 1, 2 * len(self.dates) + 1)))
        ticker_range.extend(
            list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)))
        full_hm.yaxis.ticker = ticker_range
        override_dict = dict()
        for i, date in zip(list(range(len(self.dates))), self.dates):
            override_dict[i] = date
        for i, date in zip(
                list(range(len(self.dates) + 1, 2 * len(self.dates) + 1)),
                self.dates):
            override_dict[i] = date

        for i, date in zip(
                list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)),
                self.dates):
            override_dict[i] = date
        full_hm.yaxis.major_label_overrides = override_dict
        full_hm.toolbar.active_drag = 'auto'
        full_hm.toolbar.active_scroll = "auto"
        # full_hm.rect([min(self.cds.data['xs'])+500 for __ in range(3)],
        #              [len(self.dates)/2-0.5, len(self.dates)*3/2+0.5,
        #               len(self.dates) * 5 / 2 + 1.5],
        #              color=['#FFA500', '#FFA500', '#FFA500'], width=1000,
        #              height=len(self.dates))
        full_hm.rect('xs', 'ys', color='color', source=self.background_cds,
                       width=1000, height=len(self.dates))
        # full_hm.rect('xs', 'ys', color='#00ff00', source=self.background_cds,
        #              width=1000, height=len(self.dates), fill_alpha=0.0)
        full_hm.rect('xs', 'ys', color='color', source=self.cds, width=1,
                     height=1)
        full_hm.rect('xs', 'ys', color='color', source=self.wo_cds,
                       width='width', height=1, fill_alpha=0, line_alpha=1)

        full_hm.add_tools(self.rangetool)
        full_hm.toolbar.active_multi = self.rangetool
        return full_hm

    def _build_figure(self):
        output = column([row([self.dropdown.figure], height=100),
                         row([self.yaxis_labels_2, self.full_heatmap]),
                         row([self.yaxis_labels_1, self.zoomed_heatmap,
                              self.wo_table])])
        return output

