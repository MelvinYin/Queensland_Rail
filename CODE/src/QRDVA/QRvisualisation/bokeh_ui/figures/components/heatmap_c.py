from ....utils import labels
from bokeh.plotting import figure, show, ColumnDataSource
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


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

class HeatMapComponent:
    def __init__(self, x_y_heat):
        self.x_y_heat = x_y_heat
        self.cmaps = dict()
        self.CDS = self._build_CDS()
        self.figure = self._build_figure()

    def _build_region_cmap(self):
        cmap_region = HeatmapRegionConverter()
        self.cmaps['region'] = cmap_region

    def _build_density_cmap(self):
        min_value = min(self.x_y_heat[2])
        max_value = max(self.x_y_heat[2])
        cmap_density = HeatmapColorConverter(min_value, max_value, reverse=True)
        self.cmaps['density'] = cmap_density

    def _build_CDS(self):
        self._build_region_cmap()
        initial_data = dict()
        initial_data['xs'] = self.x_y_heat[0]
        initial_data['ys'] = self.x_y_heat[1]
        initial_data['color'] = list(map(self.cmaps['region'].convert,
                                                    self.x_y_heat[2]))
        CDS = ColumnDataSource(data=initial_data)
        return CDS

    def _build_figure(self):
        p = figure(plot_width=400, plot_height=400)
        p.x("xs", 'ys', source=self.CDS, name='ys', size=2, color='color')
        return p

    def update(self, x_y_heat, cmap_type):
        """
        This cannot be CDS.stream now since previous points that are not
        replaced, are not removed.
        (obv) Points need to be dropped differently since different cols have
        different rows where they are invalid.
        CDS needs to be updated all at once otherwise a warning about
        different lengths in data dict will appear, not sure if this will
        lead to undefined behaviour.
        """
        self.x_y_heat = x_y_heat
        if cmap_type == "density":
            self._build_density_cmap()
        updated_data = dict()
        updated_data['xs'] = self.x_y_heat[0]
        updated_data['ys'] = self.x_y_heat[1]
        if cmap_type == "density":
            updated_data['color'] = list(map(self.cmaps['density'].convert,
                                             self.x_y_heat[2]))
        elif cmap_type == 'region':
            updated_data['color'] = list(
                map(self.cmaps['region'].convert, self.x_y_heat[2]))
        else:
            raise Exception
        self.CDS.data = updated_data


class HeatMapRegionComponent:
    def __init__(self, x_y_heat):
        # this should just be a tuple of 3 lists, one of X, one of Y, one of
        # heat value at each point
        self.x_y_heat = x_y_heat
        self.cmap = self._build_cmap()
        self.CDS = self._build_CDS()
        self.figure = self._build_figure()

    def _build_cmap(self):
        cmap = HeatmapRegionConverter()
        return cmap

    def _build_CDS(self):
        initial_data = dict()
        initial_data['xs'] = self.x_y_heat[0]
        initial_data['ys'] = self.x_y_heat[1]
        initial_data['color'] = list(map(self.cmap.convert, self.x_y_heat[2]))
        CDS = ColumnDataSource(data=initial_data)
        return CDS

    def _build_figure(self):
        p = figure(plot_width=400, plot_height=400)
        p.x("xs", 'ys', source=self.CDS, name='ys', size=2, color='color')
        return p


if __name__ == "__main__":
    import pickle
    import os
    from utils import paths

    with open(os.path.join(paths.DATA, "GPR_tmp.pkl"), 'rb') as file:
        df = pickle.load(file)

    x_y_heat = [df['Dec.Lat'].values[::100], df['Dec.Long'].values[::100],
                df['Sub-division'].values[::100]]
    component = HeatMapComponent(x_y_heat)
    show(component.figure)


    # df_cropped = df[df['Left*'] != 0]
    # x_y_heat = [df_cropped['Dec.Lat'].values[::10],
    #             df_cropped['Dec.Long'].values[::10],
    #             df_cropped['Left*'].values[::10]]
    # component = HeatMapComponent(x_y_heat, reverse=True)
    # df_cropped = df[df['Left*.4'] != 0]
    # x_y_heat = [df_cropped['Dec.Lat'].values[::10],
    #             df_cropped['Dec.Long'].values[::10],
    #             df_cropped['Left*.4'].values[::10]]
    # # component.update(x_y_heat)
    # show(component.figure)










"""
    # new = dict(xs=x_y_heat[0], ys=x_y_heat[1], color=list(map(
    #     component.cmap.convert, df_cropped['Left*.4'].values[::10])))
    # component.CDS.stream(new)
    # new = dict(color=[tuple([slice(len(df_cropped['Dec.Lat'].values[::10])),
    #                      list(map(component.cmap.convert, df_cropped[
    #
    'Left*.4'].values[::10]))])])
    # component.CDS.patch(new)

            new = dict(color=[tuple([slice(self.data_length),
                                 list(map(self.cmap.convert, color_data))])])
        self.CDS.patch(new)
"""
