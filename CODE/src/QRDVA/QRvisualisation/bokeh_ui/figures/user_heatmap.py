STDEV_LENGTH = 0.005

heat_width = 0.01
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, RangeTool
from .components import heatmap_c
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Select

from bokeh.plotting import figure, output_file, save
import pickle
from collections import defaultdict
import os
from utils import paths
import numpy as np


def user_trc_heatmap_preparation(input_trc_pkl=""):
    with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
        outputs = pickle.load(file)

    metrage_full = outputs['METRAGE'].drop_duplicates().values

    hm_color = heatmap_c.HeatmapColorConverter(0, 10)

    offset_left = int(round(min(metrage_full), -3) - min(metrage_full))

    num_metrage_splits = (max(metrage_full) - min(metrage_full)) // 1000

    output_by_property = list(
        [defaultdict(list) for __ in range(num_metrage_splits)])

    for split_i in range(num_metrage_splits):
        if split_i == 0:
            if offset_left > 0:
                start_metrage = min(metrage_full)
                end_metrage = offset_left + min(metrage_full)
            else:
                start_metrage = min(metrage_full)
                end_metrage = offset_left + min(metrage_full) + 1000
        else:
            if offset_left < 0:
                offset_left += 1000
            start_metrage = offset_left + min(metrage_full) + split_i * 1000
            end_metrage = offset_left + min(metrage_full) + (split_i + 1) * 1000

        selected_df = outputs[
            (outputs['METRAGE'] >= start_metrage) &
            (outputs['METRAGE'] < end_metrage)]

        top_l_data = list(
            map(hm_color.convert, selected_df['TOP_L_stdev'].values))
        top_r_data = list(
            map(hm_color.convert, selected_df['TOP_R_stdev'].values))
        tw_3_data = list(
            map(hm_color.convert, selected_df['TW_3_stdev'].values))

        metrage_local = np.array(range(start_metrage, end_metrage))

        y1 = np.array([0 for __ in range(1000)])
        y2 = np.array([0 for __ in range(1000)])
        y3 = np.array([0 for __ in range(1000)])

        assert len(metrage_local) == len(y1)
        assert len(metrage_local) == len(top_l_data)
        assert len(metrage_local) == len(top_r_data)
        assert len(metrage_local) == len(tw_3_data)

        output_by_property[split_i]['xs'].extend(list(metrage_local))
        output_by_property[split_i]['ys'].extend(list(y1))
        output_by_property[split_i]['color'].extend(list(top_l_data))

        output_by_property[split_i]['xs'].extend(list(metrage_local))
        output_by_property[split_i]['ys'].extend(list(y2))
        output_by_property[split_i]['color'].extend(list(top_r_data))

        output_by_property[split_i]['xs'].extend(list(metrage_local))
        output_by_property[split_i]['ys'].extend(list(y3))
        output_by_property[split_i]['color'].extend(list(tw_3_data))

    return output_by_property

def user_wo_heatmap_preparation(outputs):
    # def user_wo_heatmap_preparation(input_trc_pkl="C195_with_stdev.pkl"):
    #     with open(os.path.join(paths.DATA, input_trc_pkl), 'rb') as file:
    #         outputs = pickle.load(file)
    mask = outputs['predicted_output'].values == 1
    # wo_prediction_smoothed
    length = len(outputs['longitude'].values[mask])
    data_for_points = dict()
    data_for_points['xs'] = outputs['longitude'].values[mask]
    data_for_points['ys'] = outputs['latitude'].values[mask]
    min_long, max_long = min(outputs['longitude'].values), max(
        outputs['longitude'].values)
    radius = abs(max_long - min_long) * 0.0045
    data_for_points['radius'] = list([radius for __ in range(length)])
    data_for_points['color'] = list(["#964B00" for __ in range(length)])
    data_for_line = dict()
    data_for_line['xs'] = outputs['longitude'].values
    data_for_line['ys'] = outputs['latitude'].values
    return data_for_points, data_for_line

class UserWOHeatMap:
    def __init__(self, input_data, title="Work Order Prediction"):
        self.data = input_data
        self.title = title
        self.figure = self._build_figure()

    def _build_figure(self):
        p = figure(title=self.title)
        p.xaxis.axis_label = 'Longitude'
        p.yaxis.axis_label = 'Latitude'
        p.line(self.data[1]['xs'], self.data[1]['ys'], color='#0000FF')
        p.scatter(self.data[0]['xs'], self.data[0]['ys'],
                  radius=self.data[0]['radius'],
                  fill_color=self.data[0]['color'], fill_alpha=0.5,
                  line_color=None)

        return p

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

class UserTRCHeatMap:
    # todo: currently TRC uses fixed 1000 offset, but that assumes a starting
    #  position of 1000. For user trc, we want to display all, so start from
    #  whatever, and for first segment, only serve up to 1000-mark. Same as
    #  ending point. This should be done in data_preparation as well.
    def __init__(self, heatmap_dates, title=""):
        self.title = title
        self.heatmap_data = heatmap_dates
        self.starting_offset = self.heatmap_data['xs'][0]
        self.cds = self._build_cds()
        self.zoomed_heatmap = self._build_zoomed_heatmap()
        self.rangetool = self._build_rangetool()
        self.full_heatmap = self._build_full_heatmap()
        self.dropdown = self._build_dropdown()
        self.figure = self._build_figure()

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

        callback = CustomJS(args=dict(cds=self.cds,
                                      hm_data=self.heatmap_data,
                                      f_hm=self.full_heatmap,
                                      rangetool=self.rangetool,
                                      rangetool_range=rangetool_range,
                                      offset=self.starting_offset//1000),
                            code="""
        var f = cb_obj.value;
        cds.data = hm_data[f];
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
                           y_range=[5, -1],
                           tools='save')

        zoomed_hm.yaxis.major_label_text_font_size = '4pt'
        zoomed_hm.xaxis.ticker = []
        ticker_range = [0, 2, 4]
        zoomed_hm.yaxis.ticker = ticker_range
        override_dict = dict()
        override_dict[0] = "TW 3"
        override_dict[2] = "TOP L"
        override_dict[4] = "TOP R"
        zoomed_hm.yaxis.major_label_overrides = override_dict
        zoomed_hm.rect('xs', 'ys', color='color', source=self.cds, width=1,
                       height=1)
        return zoomed_hm

    def _build_full_heatmap(self):
        full_hm = figure(title=f"{self.title} Full", height=600, width=800,
                         x_range=[self.starting_offset,
                                  self.starting_offset+1000],
                         y_range=self.zoomed_heatmap.y_range,
                         tools='save')

        ticker_range = [0, 2, 4]
        full_hm.yaxis.ticker = ticker_range
        override_dict = dict()
        override_dict[0] = "TW 3"
        override_dict[2] = "TOP L"
        override_dict[4] = "TOP R"
        full_hm.yaxis.major_label_overrides = override_dict
        full_hm.toolbar.active_drag = 'auto'
        full_hm.toolbar.active_scroll = "auto"
        full_hm.rect('xs', 'ys', color='color', source=self.cds, width=1,
                     height=1)
        full_hm.add_tools(self.rangetool)
        full_hm.toolbar.active_multi = self.rangetool
        return full_hm

    def _build_figure(self):
        output = column([row([self.dropdown.figure], height=100),
                         row([self.full_heatmap]),
                         row([self.zoomed_heatmap])])
        return output

def make_user_wo_plot(input_df, output_filepath='output.html'):
    cleaned_df = user_wo_heatmap_preparation(input_df)
    hm = UserWOHeatMap(cleaned_df)
    output_file(output_filepath)
    save(hm.figure)
    return True

if __name__ == "__main__":
    pkl_path = os.path.join(paths.DATA, "test_output1_clean.pickle")
    with open(pkl_path, 'rb') as file:
        output = pickle.load(file)
    make_user_wo_plot(output)
