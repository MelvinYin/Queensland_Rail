from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.widgets import Div, RadioButtonGroup, Select
from matplotlib.colors import LinearSegmentedColormap

class RBGComponent:
    def __init__(self, specs, callback=None):
        self.specs = specs
        self.figure = self._set_RBG(callback)

    def _set_RBG(self, callback):
        RBG = RadioButtonGroup()
        RBG.width = self.specs['width']
        RBG.height = self.specs['height']
        RBG.labels = ["Region", "Left", "Centre", "Right"]
        RBG.active = None
        if callback:
            RBG.on_click(callback)
        return RBG


class DropDownComponent:
    def __init__(self, specs, callback=None):
        self.specs = specs
        self.figure = self._set_dropdown(callback)

    def _set_dropdown(self, callback):
        dropdown = Select(title=self.specs["title"], value=self.specs['menu'][
            0][0],
                          options=self.specs['menu'],
                          width=self.specs['width'],
                          height=self.specs['height'])
        if callback:
            dropdown.on_change('value', callback)
        return dropdown

class TextBoxComponent:
    def __init__(self, specs):
        self.figure = self._set_TB(specs)

    def _set_TB(self, specs):
        TB = Div(text=specs['title'])
        TB.width = specs['width']
        TB.height = specs['height']
        return TB
