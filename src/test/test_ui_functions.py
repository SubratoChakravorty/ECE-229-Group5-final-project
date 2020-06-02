import unittest
import plotly.graph_objects as go

from src.ui.dashboard import get_frequency_plot


class TestUiFunctions(unittest.TestCase):
    def test_get_frequency_plot_returns_figure(self):
        fig = get_frequency_plot(['SCH_LOCALE', 'N1HIDEG', 'SCIJOB'])
        self.assertTrue(isinstance(fig, go.Figure), f"fig is {type(fig)} not `go.Figure`")


if __name__ == '__main__':
    unittest.main()
