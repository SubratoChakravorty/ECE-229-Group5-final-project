import unittest
import plotly.graph_objects as go

from src.config import *
from src.univariate_methods import get_var_info
from src.multivariate_methods import get_feature_importance
from src.ui.dashboard import get_frequency_plot, get_sunburst_plot, make_correlation_heatmap, get_categorical_p_plot


vars_df = get_var_info(variables_file)


class TestUiFunctions(unittest.TestCase):
    def test_get_frequency_plot_returns_figure(self):
        fig = get_frequency_plot(['SCH_LOCALE', 'N1HIDEG', 'SCIJOB'])
        self.assertTrue(isinstance(fig, go.Figure), f"fig is {type(fig)} not `go.Figure`")

    def test_get_sunburst_plot_returns_figure(self):
        fig = get_sunburst_plot('X1SCIEFF', ['SCH_LOCALE', 'N1HIDEG', 'SCIJOB'])
        self.assertTrue(isinstance(fig, go.Figure), f"fig is {type(fig)} not `go.Figure`")

    def test_make_correlation_heatmap_returns_figure(self):
        fig = make_correlation_heatmap()
        self.assertTrue(isinstance(fig, go.Figure), f"fig is {type(fig)} not `go.Figure`")

    def test_get_categorical_p_plot_returns_figure(self):
        fi_dict = get_feature_importance('X1SCIEFF', ['COURSE_TYPE', 'N1GEN', 'X1GEN', 'SCH_LOCALE'], method='spearman')['categorical']
        fi_dict = {vars_df.loc[k, 'short']: v for k, v in fi_dict.items()}
        fig = get_categorical_p_plot(fi_dict)
        self.assertTrue(isinstance(fig, go.Figure), f"fig is {type(fig)} not `go.Figure`")


if __name__ == '__main__':
    unittest.main()
