import sys
import inspect
src_path = inspect.currentframe().f_code.co_filename.rsplit("/", maxsplit=2)[0]
sys.path.append(src_path)

import os
import pickle

from bokeh.plotting import show, curdoc
from bokeh.layouts import row, column

try:
    from bokeh_ui.figures import heatmap, trc_heatmap
    from bokeh_ui.figures.components import boxes_c

except ModuleNotFoundError:
    from .bokeh_ui.figures import heatmap, trc_heatmap
    from .bokeh_ui.figures.components import boxes_c
from utils import paths, labels


class GPR:
    def __init__(self):
        self.current_selection = ["LRI", 'All', 0]
        self.gpr_data = self._get_gpr_data()
        self.heatmap = self._build_heatmap()
        self.header = self._build_header()
        self.rbg = self._build_RBG()
        self.index_dropdown = self._build_index_dropdown()
        self.region_dropdown = self._build_region_dropdown()
        self.rbg_prompt = self._build_rbg_prompt()
        self.selection_column_map = self._build_select_col_map()
        self.figure = self._build_figure()

    def _build_select_col_map(self):
        selection_column_mapping = dict()

        selection_column_mapping[('LRI', 0)] = 'Sub-division'
        selection_column_mapping[('BTI', 0)] = 'Sub-division'
        selection_column_mapping[('MLI', 0)] = 'Sub-division'
        selection_column_mapping[('SMLI', 0)] = 'Sub-division'
        selection_column_mapping[('FDL', 0)] = 'Sub-division'
        selection_column_mapping[('CTQI', 0)] = 'Sub-division'

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
        return selection_column_mapping

    def region_callback(self, attr, old, new):
        self.current_selection[1] = new
        self._update_figure()

    def index_callback(self, attr, old, new):
        self.current_selection[0] = new
        self._update_figure()

    def rbg_callback(self, new):
        self.current_selection[2] = int(new)
        self._update_figure()

    def _update_figure(self):
        if None in self.current_selection:
            return
        column_key = tuple([self.current_selection[0],
                            self.current_selection[2]])
        assert column_key in self.selection_column_map
        column_name = self.selection_column_map[column_key]
        self.heatmap.update((column_name, self.current_selection[1]))

    def _build_region_dropdown(self):
        menu = list(tuple([state, state]) for state in labels.states)
        menu.insert(0, ("All", "All"))
        specs = dict(width=100, height=50, menu=menu, title="Region")
        dropdown = boxes_c.DropDownComponent(specs, self.region_callback)
        return dropdown

    def _build_index_dropdown(self):
        menu = [("LRI", "LRI"), ("BTI", "BTI"), ("MLI", "MLI"),
                ("SMLI", "SMLI"), ("FDL", "FDL"), ("CTQI", "CTQI")]
        specs = dict(width=100, height=50, menu=menu, title="Metric")
        dropdown = boxes_c.DropDownComponent(specs, self.index_callback)
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
        rbg = boxes_c.RBGComponent(specs, self.rbg_callback)
        return rbg

    def _build_figure(self):
        header_row = row(self.header.figure, height=50)
        rbg_col = column([self.rbg_prompt.figure, self.rbg.figure])
        dropdown_row = row([row(self.index_dropdown.figure, width=150),
                            self.region_dropdown.figure], width=150)
        selection_row = row(dropdown_row, height=100)
        rbg_row = row(rbg_col, height=100)
        heatmap_row = row(self.heatmap.figure, height=450)
        figure = column([header_row, selection_row, rbg_row, heatmap_row])
        return figure

    def _get_gpr_data(self):
        with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
            df = pickle.load(file)
        return df

    def _build_heatmap(self):
        hm = heatmap.HeatMap(self.gpr_data)
        return hm


class UI:
    def __init__(self):
        self.gpr = self._build_gpr()
        self.project_header = self._build_p_header()
        self.project_description = self._build_p_descr()
        self.trc_heatmap = self._build_trc_heatmap()
        self.figure = self._build_figure()

    def _build_trc_heatmap(self):
        with open(os.path.join(paths.DATA_DERIVED, "trc_heatmap_data.pkl"),
                  'rb') as file:
            heatmap_data = pickle.load(file)
        hm = trc_heatmap.TRCHeatMap(heatmap_data)
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


def show_plot():
    ui = UI()
    #curdoc().add_root(ui.figure)
    return ui.figure

ui = UI()
curdoc().add_root(ui.figure)
show(ui.figure)
