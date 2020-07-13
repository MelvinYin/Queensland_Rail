import sys
import inspect
src_path = inspect.currentframe().f_code.co_filename.rsplit("/", maxsplit=2)[0]
sys.path.append(src_path)

import os
import pickle

import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

from bokeh.document import Document
from bokeh.layouts import row, column
from bokeh.models.widgets import Button
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import HTMLTemplateFormatter, DataTable, TableColumn, RadioButtonGroup, \
    Select
from bokeh.plotting import figure, ColumnDataSource

try:
    from QRDVA.QRvisualisation.bokeh_ui.figures import heatmap, trc_heatmap
    from QRDVA.QRvisualisation.bokeh_ui.figures.components import boxes_c
    from utils import paths, labels
except ModuleNotFoundError:
    from .figures import heatmap, trc_heatmap
    from .figures.components import boxes_c
    from ..utils import paths, labels


class GPR:
    def __init__(self):
        self.gpr_data = self._get_gpr_data()
        self.CDS = self._build_CDS()
        self.header = self._build_header()
        self.rbg_prompt = self._build_rbg_prompt()
        self.state_table = self._build_state_color_table()
        self.heatmap = self._build_heatmap()

        self.rbg = self._build_RBG()
        self.metric_dropdown = self._build_index_dropdown()
        self.region_dropdown = self._build_region_dropdown()

        self._add_rbg_callback()
        self._add_metric_callback()
        self._add_region_callback()
        self.figure = self._build_figure()

    def _build_state_color_table(self):
        template = """
        <div style="background:<%= value[0] %>; 
            color: white"> 
        <%= "_"%></div>
        """
        formatter = HTMLTemplateFormatter(template=template)
        converter = HeatmapRegionConverter()
        states = list(converter.state_color_map.keys())
        colors = list(converter.state_color_map.values())
        i_sort = np.argsort(states)
        states = np.array(states)[i_sort]
        colors = np.array(colors)[i_sort]
        cds_input = dict()
        cds_input['region'] = states
        cds_input['color'] = []
        colors_str = [" " for __ in range(len(states))]
        for color_code, color_str in zip(colors, colors_str):
            color_hex = "#%02x%02x%02x" % (
                int(color_code[0] * 255), int(color_code[1] * 255),
                int(color_code[2] * 255))
            cds_input['color'].append((color_hex, color_str))
        cds = ColumnDataSource(cds_input)
        columns = [
            TableColumn(field="region", title="Region", width=60),
            TableColumn(field="color", title="Colour", formatter=formatter,
                        width=50)]
        table = DataTable(source=cds, columns=columns, height=350, width=200,
                          fit_columns=True, sortable=False, reorderable=False,
                          selectable=False, index_position=None)
        return table

    def _add_region_callback(self):
        rbg_label_map = ['region', 'left', 'centre', 'right']
        callback = CustomJS(
            args=dict(cds=self.CDS, gpr_data=self.gpr_data,
                      hm=self.heatmap,
                      rbg_labels=rbg_label_map, rbg_obj=self.rbg,
                      metric_obj=self.metric_dropdown),
        code="""
        var f = rbg_obj.active;
        var name = rbg_labels[f];
        if (name != "region"){
            var region = cb_obj.value;
            var metric = metric_obj.value;
            cds.data['xs'] = gpr_data['xs'][region];
            cds.data['ys'] = gpr_data['ys'][region];
            cds.data['color'] = gpr_data[name][region][metric];
        }
        cds.change.emit();
        """)
        self.region_dropdown.js_on_change('value', callback)

    def _add_rbg_callback(self):
        rbg_label_map = ['region', 'left', 'centre', 'right']
        callback = CustomJS(
            args=dict(cds=self.CDS, gpr_data=self.gpr_data, hm=self.heatmap,
                      rbg_labels=rbg_label_map,
                      metric_obj=self.metric_dropdown,
                      region_obj=self.region_dropdown), code="""
        var f = cb_obj.active;
        var name = rbg_labels[f];
        var metric = metric_obj.value;
        var region = region_obj.value;
        if (name == "region"){
            cds.data['xs'] = gpr_data[name]['xs'];
            cds.data['ys'] = gpr_data[name]['ys'];
            cds.data['color'] = gpr_data[name]['color'];
        } else {
            cds.data['xs'] = gpr_data['xs'][region];
            cds.data['ys'] = gpr_data['ys'][region];
            cds.data['color'] = gpr_data[name][region][metric];
        }
        cds.change.emit();
        """)
        self.rbg.js_on_click(callback)

    def _add_metric_callback(self):
        rbg_label_map = ['region', 'left', 'centre', 'right']
        callback = CustomJS(
            args=dict(cds=self.CDS, gpr_data=self.gpr_data, hm=self.heatmap,
                      rbg_labels=rbg_label_map, rbg_obj=self.rbg,
                      region_obj=self.region_dropdown), code="""
        var f = rbg_obj.active;
        var name = rbg_labels[f];
        if (name != "region"){
            var metric = cb_obj.value;
            var region = region_obj.value;
            cds.data['xs'] = gpr_data['xs'][region];
            cds.data['ys'] = gpr_data['ys'][region];
            cds.data['color'] = gpr_data[name][region][metric];
        }
        cds.change.emit();
        """)
        self.metric_dropdown.js_on_change('value', callback)

    def _build_CDS(self):
        CDS = ColumnDataSource(data=self.gpr_data['region'])
        return CDS

    def _build_region_dropdown(self):
        menu = list(tuple([state, state]) for state in labels.states)
        specs = dict(width=100, height=50, menu=menu, title="Region")
        dropdown = Select(title=specs["title"],
                                     value=specs['menu'][0][0],
                                     options=specs['menu'],
                                     width=specs['width'],
                                     height=specs['height'])
        return dropdown

    def _build_index_dropdown(self):
        menu = [("LRI", "LRI"), ("BTI", "BTI"), ("MLI", "MLI"),
                ("SMLI", "SMLI"), ("FDL", "FDL"), ("CTQI", "CTQI")]
        specs = dict(width=100, height=50, menu=menu, title="Metric")
        dropdown = Select(title=specs["title"],
                         value=specs['menu'][0][0],
                         options=specs['menu'],
                         width=specs['width'],
                         height=specs['height'])
        return dropdown

    def _build_rbg_prompt(self):
        specs = dict(title="Track", height=1, width=150)
        prompt = boxes_c.TextBoxComponent(specs)
        return prompt

    def _build_header(self):
        specs = dict(title="Mapped QR 2018 GPR Metrics", height=10, width=500)
        header = boxes_c.TextBoxComponent(specs)
        return header

    def _build_RBG(self):
        specs = dict(width=250, height=30)
        RBG = RadioButtonGroup()
        RBG.width = specs['width']
        RBG.height = specs['height']
        RBG.labels = ["Region", "Left", "Centre", "Right"]
        RBG.active = 0
        return RBG

    def _build_figure(self):
        header_row = row(self.header.figure, height=50)
        rbg_col = column([self.rbg_prompt.figure, self.rbg])
        dropdown_row = row([row(self.metric_dropdown, width=150),
                            self.region_dropdown], width=150)
        selection_row = row(dropdown_row, height=100)
        rbg_row = row(rbg_col, height=100)
        heatmap_row = row(self.heatmap, height=450, width=450)
        table_row = row(self.state_table, height=450)
        heatmap_table = row([heatmap_row, table_row])
        figure = column([header_row, selection_row, rbg_row, heatmap_table])
        return figure

    def _get_gpr_data(self):
        with open(os.path.join(paths.DATA, "gpr_tmp5.pkl"), 'rb') as file:
            df = pickle.load(file)
        return df

    def _build_heatmap(self):
        hm = figure(plot_width=400, plot_height=400)
        hm.x("xs", 'ys', source=self.CDS, name='ys', size=2, color='color')
        hm.xaxis.axis_label = "Latitude"
        hm.yaxis.axis_label = "Longitude"
        return hm


class ButtonURLComponent:
    def __init__(self, specs):
        self.widget = self._set_button(specs)

    def _url_callback(self):
        url = "trc_heatmap.html"
        args = dict(url=url)
        obj = CustomJS(args=args, code='window.open(url);')
        return obj

    def _set_button(self, specs):
        button = Button()
        button.label = specs['text']
        button.width = specs['width']
        button.height = specs['height']
        button.name = os.path.join(paths.DATA, "trc_heatmap.html")
        button.callback = self._url_callback()
        return button

class UI:
    def __init__(self):
        self.gpr = self._build_gpr()
        self.project_header = self._build_p_header()
        self.project_description = self._build_p_descr()
        self.trc_heatmap = self._build_trc_heatmap()
        self.figure = self._build_figure()

    def _build_trc_heatmap(self):
        with open(os.path.join(paths.DATA, "C195_input.pkl"), 'rb') as file:
            heatmap_data = pickle.load(file)
        if not os.path.isfile(os.path.join(paths.DATA, "wo_input.pkl")):
            trc_heatmap.create_wo_input()
        with open(os.path.join(paths.DATA, "wo_input.pkl"), 'rb') as file:
            wo_input = pickle.load(file)
        if not os.path.isfile(os.path.join(paths.DATA, "wo_type_map.pkl")):
            trc_heatmap.wo_type_map_data_preparation()
        with open(os.path.join(paths.DATA, "wo_type_map.pkl"), 'rb') as file:
            wo_type_map = pickle.load(file)
        title = "TRC Standard Deviation"
        hm = trc_heatmap.TRCHeatMap(heatmap_data, title, wo_input, wo_type_map)
        return hm

    def _build_gpr(self):
        gpr = GPR()
        return gpr

    def _build_p_descr(self):
        specs = dict(title="Project_description", height=0, width=500)
        descr = boxes_c.TextBoxComponent(specs)
        return descr

    def _build_p_header(self):
        specs = dict(title="DVA Project", height=0, width=500)
        header = boxes_c.TextBoxComponent(specs)
        return header

    def _build_figure(self):
        header_row = row(self.project_header.figure, height=20)
        description_row = row(self.project_description.figure, height=20)
        gpr_row = row(self.gpr.figure)
        trc_hm_row = row(self.trc_heatmap.figure)
        figure = column([header_row, description_row, gpr_row, trc_hm_row])
        return figure



class HeatmapRegionConverter:
    def __init__(self):
        self.state_color_map = self._create_cmap()

    def _create_cmap(self):
        cmap = plt.get_cmap('nipy_spectral')
        state_color_map = dict()
        for i, state in enumerate(labels.states):
            cmap_value = i * 255 / len(labels.states)
            color = cmap(int(cmap_value))
            state_color_map[state] = color
        return state_color_map

    def convert(self, state):
        assert state in self.state_color_map
        color_rgb = self.state_color_map[state]
        color_hex = "#%02x%02x%02x" % (
            int(color_rgb[0] * 255), int(color_rgb[1] * 255),
            int(color_rgb[2] * 255))
        return color_hex


class HeatmapColorConverter:
    def __init__(self, min_value, max_value, reverse=True):
        self.min_value = min_value
        self.max_value = max_value
        self.reverse = reverse
        self._cmap = self._create_cmap()

    def _create_cmap(self):
        c = ["gray", "orange"]
        v = [0, 1.]
        l = list(zip(v, c))
        cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
        return cmap

    def convert(self, value):
        if self.reverse:
            value = self.max_value - value
        else:
            value = value - self.min_value
        value /= (self.max_value - self.min_value)
        color_rgb = self._cmap(value)
        color_hex = "#%02x%02x%02x" % (
            int(color_rgb[0] * 255), int(color_rgb[1] * 255),
            int(color_rgb[2] * 255))
        return color_hex


def show_plot():
    with open(os.path.join(paths.DATA, "ui_cached.json"), 'r') as file:
        json_str = json.load(file)
    document = Document.from_json_string(json_str)
    return document.roots[0]

if __name__ == "__main__":
    # show(UI().gpr.figure)
    # doc = Document()
    # doc.add_root(UI().figure)
    # doc_json = doc.to_json()
    # json_string = json.dumps(doc_json)
    # with open(os.path.join(paths.DATA, "ui_cached.json"), 'w') as file:
    #     json.dump(json_string, file)
    pass
    # doc = Document()
    # doc.add_root(UI().figure)
    # doc_json = doc.to_json()
    # json_string = json.dumps(doc_json)
    # with open(os.path.join(paths.DATA, "ui_cached.json"), 'w') as file:
    #     json.dump(json_string, file)