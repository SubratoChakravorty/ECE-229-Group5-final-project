import unittest

import plotly.graph_objects as go

from src.config import *
from src.univariate_methods import get_var_info
from src.multivariate_methods import get_feature_importance, MLmodel
from src.ui.dashboard import get_frequency_plot, get_sunburst_plot, make_correlation_heatmap, get_categorical_p_plot, \
    make_prediction_plot, make_importance_bar_plot, make_hist_plot, update_text, make_categorical_importance_plots, \
    train_model


# unwrap callbacks
make_prediction_plot = make_prediction_plot.__wrapped__
make_importance_bar_plot = make_importance_bar_plot.__wrapped__
make_hist_plot = make_hist_plot.__wrapped__
update_text = update_text.__wrapped__
make_categorical_importance_plots = make_categorical_importance_plots.__wrapped__

vars_df = get_var_info(variables_file)


class TestUiFunctions(unittest.TestCase):
    def test_get_frequency_plot_returns_figure(self):
        fig = get_frequency_plot(['SCH_LOCALE', 'N1HIDEG', 'SCIJOB'])
        self.assertIsInstance(fig, go.Figure, f"fig is {type(fig)} not `go.Figure`")

    def test_get_sunburst_plot_returns_figure(self):
        fig = get_sunburst_plot('X1SCIEFF', ['SCH_LOCALE', 'N1HIDEG', 'SCIJOB'])
        self.assertIsInstance(fig, go.Figure, f"fig is {type(fig)} not `go.Figure`")

    def test_make_correlation_heatmap_returns_figure(self):
        fig = make_correlation_heatmap()
        self.assertIsInstance(fig, go.Figure, f"fig is {type(fig)} not `go.Figure`")

    def test_get_categorical_p_plot_returns_figure(self):
        fi_dict = get_feature_importance('X1SCIEFF', ['COURSE_TYPE', 'N1GEN', 'X1GEN', 'SCH_LOCALE'], method='spearman')['categorical']
        fi_dict = {vars_df.loc[k, 'short']: v for k, v in fi_dict.items()}
        fig = get_categorical_p_plot(fi_dict)
        self.assertIsInstance(fig, go.Figure, f"fig is {type(fig)} not `go.Figure`")

    def test_make_prediction_plot_returns_sunburst_if_exog_empty(self):
        slider_values = 0., 0., 0., 0.
        exog = []
        endog = 'item1'
        x_var = 'item1'
        fig = make_prediction_plot(exog, endog, x_var, slider_values)[0]
        trace = next(fig.select_traces())
        self.assertIsInstance(trace, go.Sunburst)

    def test_make_prediction_plot_returns_sunburst_if_endog_empty(self):
        slider_values = 0., 0., 0., 0.
        exog = ['item1', 'item2']
        endog = None
        x_var = 'item1'
        fig = make_prediction_plot(exog, endog, x_var, slider_values)[0]
        trace = next(fig.select_traces())
        self.assertIsInstance(trace, go.Sunburst)

    def test_make_prediction_plot_returns_sunburst_if_x_var_empty(self):
        slider_values = 0., 0., 0., 0.
        exog = ['item1', 'item2']
        endog = 'item1'
        x_var = None
        fig = make_prediction_plot(exog, endog, x_var, slider_values)[0]
        trace = next(fig.select_traces())
        self.assertIsInstance(trace, go.Sunburst)

    def test_make_importance_bar_plot_returns_sunburst_if_exog_empty(self):
        exog = []
        endog = 'item1'
        fig = make_importance_bar_plot(exog, endog)
        trace = next(fig.select_traces())
        self.assertIsInstance(trace, go.Sunburst)

    def test_make_importance_bar_plot_returns_sunburst_if_endog_empty(self):
        exog = ['item1', 'item2']
        endog = None
        fig = make_importance_bar_plot(exog, endog)
        trace = next(fig.select_traces())
        self.assertIsInstance(trace, go.Sunburst)

    def test_make_hist_plot_returns_dict_or_figure_if_fields_empty(self):
        fields = []
        bar_width = 1
        fig = make_hist_plot(fields, bar_width)
        self.assertIsInstance(fig, (dict, go.Figure))

    def test_update_text_returns_tuple_of_strings_if_data_empty(self):
        data = []
        txt = update_text(data)
        for s in txt:
            self.assertIsInstance(s, str)
        self.assertIsInstance(txt, tuple)

    def test_make_categorical_importance_plots_returns_sunbursts_if_exog_is_empty(self):
        exog = []
        endog = 'item'
        tup = make_categorical_importance_plots(exog, endog)
        self.assertIsInstance(tup, tuple)
        self.assertEqual(2, len(tup))
        for fig in tup:
            trace = next(fig.select_traces())
            self.assertIsInstance(trace, go.Sunburst)

    def test_make_categorical_importance_plots_returns_sunbursts_if_endog_is_empty(self):
        exog = ['item1', 'item2']
        endog = None
        tup = make_categorical_importance_plots(exog, endog)
        self.assertIsInstance(tup, tuple)
        self.assertEqual(2, len(tup))
        for fig in tup:
            trace = next(fig.select_traces())
            self.assertIsInstance(trace, go.Sunburst)

    def test_train_model_returns_ml_model(self):
        model = train_model('X1SCIEFF', ['X1GEN'], 'X1SES')
        self.assertIsInstance(model, MLmodel)


if __name__ == '__main__':
    unittest.main()
