import pickle

STDEV_LENGTH = 0.005
import sys
import inspect

src_path = inspect.currentframe().f_code.co_filename.rsplit("/", maxsplit=3)[0]
sys.path.append(src_path)
print(src_path)

import os
import copy

from bokeh.plotting import figure, show, curdoc
heat_width = 0.01
from matplotlib.colors import LinearSegmentedColormap
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, RangeTool
from utils import paths
from collections import defaultdict
from bokeh_ui.figures.components import heatmap_c, boxes_c
from bokeh.models.callbacks import CustomJS

import numpy as np

def trc_heatmap_data_preparation(input_trc_pkl="C138_with_stdev.pkl"):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)

    metrage_full = outputs['METRAGE'].drop_duplicates().values

    hm_color = heatmap_c.HeatmapColorConverter(
        min(outputs['TOP_L_stdev'].values),
        max(outputs['TOP_L_stdev'].values)-10)
    print(max(outputs['TOP_L_stdev'].values))
    offset_left = int(round(min(metrage_full), -3) - min(metrage_full))
    num_metrage_splits = (max(metrage_full) - min(metrage_full))//1000
    num_dates = len(np.unique(outputs['Date'].values))

    output_by_property = list(
        [defaultdict(list) for __ in range(num_metrage_splits)])

    dates = []

    for split_i in range(num_metrage_splits):
        start_metrage = offset_left + min(metrage_full) + split_i * 1000
        end_metrage = offset_left + min(metrage_full) + (split_i+1) * 1000
        selected_df_all_dates = outputs[(outputs['METRAGE'] >= start_metrage)
                                        & (outputs['METRAGE'] < end_metrage)]
        for date_i, (date, selected_df) in \
                enumerate(selected_df_all_dates.groupby("Date")):
            if len(dates) != num_dates:
                dates.append(date)
            metrage_local = range(start_metrage, end_metrage)
            y1 = list([date_i for __ in range(1000)])
            y2 = list([date_i + num_dates + 1 for __ in range(1000)])
            y3 = list([date_i + 2*num_dates + 2 for __ in range(1000)])

            top_l_data = list(map(hm_color.convert,
                                  selected_df['TOP_L_stdev']))
            top_r_data = list(map(hm_color.convert,
                                  selected_df['TOP_R_stdev']))
            tw_3_data = list(map(hm_color.convert,
                                 selected_df['TW_3_stdev']))

            assert len(metrage_local) == len(y1)
            assert len(metrage_local) == len(top_l_data)
            assert len(metrage_local) == len(top_r_data)
            assert len(metrage_local) == len(tw_3_data)

            output_by_property[split_i]['xs'].extend(metrage_local)
            output_by_property[split_i]['ys'].extend(y1)
            output_by_property[split_i]['color'].extend(top_l_data)

            output_by_property[split_i]['xs'].extend(metrage_local)
            output_by_property[split_i]['ys'].extend(y2)
            output_by_property[split_i]['color'].extend(top_r_data)

            output_by_property[split_i]['xs'].extend(metrage_local)
            output_by_property[split_i]['ys'].extend(y3)
            output_by_property[split_i]['color'].extend(tw_3_data)

    return output_by_property, dates


from bokeh.models.widgets import Div, RadioButtonGroup, Select

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
    def __init__(self, heatmap_dates, title=""):
        self.title = title
        self.heatmap_data = heatmap_dates[0]
        self.dates = heatmap_dates[1]
        self.cds = self._build_cds()
        self.zoomed_heatmap = self._build_zoomed_heatmap()
        self.rangetool = self._build_rangetool()
        self.full_heatmap = self._build_full_heatmap()
        self.dropdown = self._build_dropdown()
        self.yaxis_labels_1 = self._build_yaxis_labels_1()
        self.yaxis_labels_2 = self._build_yaxis_labels_2()
        self.figure = self._build_figure()

    def _build_yaxis_labels_1(self):
        tb0_specs = dict(title="", width=20, height=70)
        tb0 = boxes_c.TextBoxComponent(tb0_specs)
        tb1_specs = dict(title="TOP_L", width=20, height=170)
        tb1 = boxes_c.TextBoxComponent(tb1_specs)
        tb2_specs = dict(title="TOP_R", width=20, height=170)
        tb2 = boxes_c.TextBoxComponent(tb2_specs)
        tb3_specs = dict(title="TW_3", width=20, height=170)
        tb3 = boxes_c.TextBoxComponent(tb3_specs)
        label_column = column([tb0.figure, tb1.figure, tb2.figure, tb3.figure],
                              width=70, height=200)
        return label_column

    def _build_yaxis_labels_2(self):
        tb0_specs = dict(title="", width=20, height=75)
        tb0 = boxes_c.TextBoxComponent(tb0_specs)
        tb1_specs = dict(title="TOP_L", width=20, height=170)
        tb1 = boxes_c.TextBoxComponent(tb1_specs)
        tb2_specs = dict(title="TOP_R", width=20, height=170)
        tb2 = boxes_c.TextBoxComponent(tb2_specs)
        tb3_specs = dict(title="TW_3", width=20, height=180)
        tb3 = boxes_c.TextBoxComponent(tb3_specs)
        label_column = column([tb0.figure, tb1.figure, tb2.figure,
                               tb3.figure],
                              width=70, height=200)
        return label_column

    def _build_dropdown(self):
        dropdown_specs = dict()
        dropdown_specs["title"] = "Metrage Range Selection"
        dropdown_specs['menu'] = [(str(i), f"{i*1000+11000}-"
                                           f"{(i+1)*1000+11000}") for i in
                                  range(len(self.heatmap_data))]
        dropdown_specs['width'] = 100
        dropdown_specs['height'] = 100
        rangetool_range = self.rangetool.x_range.end - self.rangetool.x_range.start

        from bokeh.models.callbacks import CustomJS

        callback = CustomJS(args=dict(cds=self.cds,
                                      hm_data=self.heatmap_data,
                                      f_hm=self.full_heatmap,
                                      rangetool=self.rangetool,
                                      rangetool_range=rangetool_range),
                            code="""
        var f = cb_obj.value;
        cds.data = hm_data[f];
        f_hm.x_range.start = (Number(f)+11)*1000;
        f_hm.x_range.end = (Number(f)+12)*1000;
        rangetool.x_range.start = (Number(f)+11) * 1000;
        rangetool.x_range.end = (Number(f)+11) * 1000 + rangetool_range;
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
        zoomed_hm = figure(title=f"{self.title} Zoomed",
                           plot_height=600,
                           plot_width=500,
                           x_axis_location="below", background_fill_color="#efefef",
                   x_range=(11000, 11050), y_range=[41, -1], tools='save')
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
        zoomed_hm.rect('xs', 'ys', color='color', source=self.cds, width=1, height=1)
        return zoomed_hm

    def _build_full_heatmap(self):
        full_hm = figure(title=f"{self.title} Full", height=600,
                         width=1000,
                         x_range=[11000, 12000],
                         y_range=self.zoomed_heatmap.y_range,
                         tools='reset,save')

        ticker_range = []
        ticker_range.extend(list(range(len(self.dates))))
        ticker_range.extend(list(range(len(self.dates)+1, 2*len(self.dates)+1)))
        ticker_range.extend(
            list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)))
        full_hm.yaxis.ticker = ticker_range
        override_dict = dict()
        for i, date in zip(list(range(len(self.dates))), self.dates):
            override_dict[i] = date
        for i, date in zip(list(range(len(self.dates) + 1,
                                      2 * len(self.dates) + 1)), self.dates):
            override_dict[i] = date

        for i, date in zip(
                list(range(2 * len(self.dates) + 2, 3 * len(self.dates) + 2)),
                self.dates):
            override_dict[i] = date
        full_hm.yaxis.major_label_overrides = override_dict
        full_hm.toolbar.active_drag = 'auto'
        full_hm.toolbar.active_scroll = "auto"
        full_hm.rect('xs', 'ys', color='color', source=self.cds, width=1, height=1)
        full_hm.add_tools(self.rangetool)
        full_hm.toolbar.active_multi = self.rangetool
        return full_hm

    def _build_figure(self):
        output = column([self.dropdown.figure,
                         row([self.yaxis_labels_1, self.zoomed_heatmap]),
                         row([self.yaxis_labels_2, self.full_heatmap])])
        return output


# heatmap_data = trc_heatmap_data_preparation()
# with open(os.path.join(paths.DATA, "trc_heatmap_data.pkl"),
#           'wb') as file:
#     pickle.dump(heatmap_data, file, -1)

# with open(os.path.join(paths.DATA, "trc_heatmap_data.pkl"), 'rb') as file:
#     heatmap_data = pickle.load(file)
#
# title = "TRC Standard Deviation"
# hm = TRCHeatMap((heatmap_data[0], heatmap_data[1]), title)
# # print(heatmap_data[0]['TOP_L_stdev'])
# show(hm.figure)
# curdoc().add_root(hm.figure)

