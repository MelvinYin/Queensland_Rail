from .components import heatmap_c
from ...utils import paths, labels


class HeatMap:
    def __init__(self, gpr_data):
        self.gpr_data = gpr_data
        self.heatmap_c = self._build_heatmap_c()
        self.figure = self._build_figure()

    def _build_heatmap_c(self):
        x_y_heat = [self.gpr_data['Dec.Lat'].values[::100],
                    self.gpr_data['Dec.Long'].values[::100],
                    self.gpr_data['Sub-division'].values[::100]]
        component = heatmap_c.HeatMapComponent(x_y_heat)
        return component

    def update(self, update_key):
        column_name, region = update_key
        assert column_name in set(self.gpr_data.columns)
        if region == "All":
            if column_name == "Sub-division":
                x_y_heat = [self.gpr_data['Dec.Lat'].values[::50],
                            self.gpr_data['Dec.Long'].values[::50],
                            self.gpr_data[column_name].values[::50]]
                self.heatmap_c.update(x_y_heat, "region")
            else:
                gpr_cropped = self.gpr_data[self.gpr_data[column_name] != 0]
                x_y_heat = [gpr_cropped['Dec.Lat'].values[::50],
                            gpr_cropped['Dec.Long'].values[::50],
                            gpr_cropped[column_name].values[::50]]
                self.heatmap_c.update(x_y_heat, "density")
        else:
            assert region in labels.states
            gpr_subdivision = self.gpr_data[self.gpr_data["Sub-division"] ==
                                            region]
            if column_name == "Sub-division":
                x_y_heat = [gpr_subdivision['Dec.Lat'].values,
                            gpr_subdivision['Dec.Long'].values,
                            gpr_subdivision[column_name].values]
                self.heatmap_c.update(x_y_heat, "region")
            else:
                gpr_cropped = gpr_subdivision[gpr_subdivision[column_name] != 0]
                x_y_heat = [gpr_cropped['Dec.Lat'].values,
                            gpr_cropped['Dec.Long'].values,
                            gpr_cropped[column_name].values]
                self.heatmap_c.update(x_y_heat, "density")
        self.figure = self._build_figure()

    def _build_figure(self):
        fig = self.heatmap_c.figure
        return fig

if __name__ == "__main__":
    import os
    import pickle

    from bokeh.plotting import show

    with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
        df = pickle.load(file)
    hm = HeatMap(df)
    hm.update(['Left*.4', "All"])
    show(hm.figure)
