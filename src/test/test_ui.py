import unittest
import re

import pandas as pd

import src.config
from src.ui import app

category_df = pd.read_csv(src.config.variables_file)


def test_ui001_sunburst_plot(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_contains_text('#sunburst_plot', "Select a category", timeout=2)

    dash_duo.select_dcc_dropdown('#expl_category_selector', index=1)
    dash_duo.wait_for_contains_text('#sunburst_plot', "Select a score", timeout=2)

    dash_duo.select_dcc_dropdown('#expl_continuous_selector', index=1)
    dash_duo.wait_for_contains_text('#sunburst_plot', "\nmean", timeout=2)


def test_ui002_second_univariate_plot_frequency(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_contains_text('#plot_selector', "frequency plot", timeout=2)

    dash_duo.select_dcc_dropdown('#expl_category_selector', index=1)
    dash_duo.wait_for_contains_text('#second_explore_plot', "\ncount", timeout=2)


def test_ui003_second_univariate_plot_box_defaults(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_contains_text('#plot_selector', "frequency plot", timeout=2)

    dash_duo.select_dcc_dropdown('#expl_category_selector', index=1)
    dash_duo.select_dcc_dropdown('#plot_selector', index=0)
    dash_duo.wait_for_contains_text('#second_explore_plot', "select a score", timeout=2)


def test_ui003_second_univariate_plot_box(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_contains_text('#plot_selector', "frequency plot", timeout=2)
    dash_duo.select_dcc_dropdown('#expl_category_selector', index=1)
    dash_duo.select_dcc_dropdown('#plot_selector', index=0)
    dash_duo.select_dcc_dropdown('#expl_continuous_selector', index=1)

    short_text = dash_duo.find_element('#expl_continuous_selector').text
    short_text, _ = re.compile(r"[^a-zA-Z ]").subn('', short_text)
    name = category_df.loc[category_df['short'] == short_text, 'name'].item()
    dash_duo.wait_for_contains_text('#second_explore_plot', name, timeout=3)

def test_ui004_histogram_plot(dash_duo):
    dash_duo.start_server(app)
    dash_duo.select_dcc_dropdown('#continuous_selector', index=1)
    dash_duo.wait_for_element('#width_slider', timeout=2)
    dash_duo.wait_for_element('#hist_plot', timeout=2)

def test_ui005_four_blocks(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_element('#info-container', timeout=2)

def test_ui006_report(dash_duo):
    dash_duo.start_server(app)
    dash_duo.wait_for_element('#open-xl', timeout=2)
    dash_duo.wait_for_element('#report', timeout=2)

if __name__ == '__main__':
    unittest.main()
