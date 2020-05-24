"""
Just run using `python dashboard.py`
"""
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from src.config import variables_file, student_data_file
from src.univariate_methods import get_hierarchical_data, get_var_info, get_field_data, get_binned_data

# # Style configuration
# external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app.css.append_css({"external_url":external_css})

# color for frontend
colors = {
    'background': '#111111',
    'text'      : '#7FDBFF'
}

plot_lookup = {0: 'box plot',
               1: 'frequency plot'}

# Populate fields from data
vars_df = get_var_info(variables_file)

report_text = """
 __   __    _______    _______    ______  
|  | |  |  |       |  |       |  |      | 
|  | |  |  |       |  |  _____|  |  _    |
|  |_|  |  |       |  | |_____   | | |   |
|       |  |      _|  |_____  |  | |_|   |
|       |  |     |_    _____| |  |       |
|_______|  |_______|  |_______|  |______| 

    Report

                                        """


def populate_dropdown(category: str):
    assert category in ['continuous', 'categorical'], f"category must be 'continuous' or 'categorical', not {category}"
    df = vars_df.loc[vars_df['type'] == category, 'short']
    return [dict(label=v, value=k) for k, v in df.to_dict().items()]


app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create app layout
app.layout = html.Div(
    [
        ######################################################< TOP PART >##################################################
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("ucsd-logo.png"),
                            id="plotly-image",
                            style={
                                "height"       : "60px",
                                "width"        : "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Analysis of Ninth Grader’s Science Self Efficacy",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "ECE229 - Team 5", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="two-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("github", id="learn-more-button"),
                            href="https://github.com/SubratoChakravorty/ECE-229-Group5-final-project",
                        )
                    ],
                    className="three-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        ######################################################< TAG1 PART >##################################################

        html.Div(
            [
                html.H4(className='how', children="How to use this dashboard"),

                html.P('1. Explore'
                       'In this section, feel free to explore the distribution of data. We provide pie chart, xx chart, ...etc.'),
                html.P(
                    '2. Inspect'
                    'We provide univariate analysis in this part. You can find how single variable affects your self-efficiency.'),
                html.P(
                    '3. Insights'
                    'Multivariate statistical analysis can be found here, which helps you to gain insights on the data and provides advice for yourself.'),

                html.H4(className='Dataset', children="Dataset"),
                html.P(
                    'This study employs public-use data from the High School Longitudinal Study of 2009 (HSLS:09). One important difference'
                    'between HSLS:09 and previous studies is its focus on STEM education; one specific goal of the study is to gain an '
                    'understanding of the factors that lead students to choose science, technology, engineering, and mathematics courses, majors, and careers.'),

                html.P("Dataset can be downloaded by clicking: "),
                html.Div([
                    'Reference: ',
                    html.A('Dataset',
                           href='https://nces.ed.gov/EDAT/Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip)')
                ]),
                html.H4(className='author', children="Author"),

                html.Br()
            ],
            className="pretty_container",
            style={"margin-bottom": "25px"}
        ),

        # ##############################################< TAG2 PART >############################################

        html.Div(
            [
                html.Div(
                    [
                        html.H1("Explore"),
                        html.P("Click a category on the inner plot to filter"),
                        html.P(["Select categories:",
                                dcc.Dropdown(id='expl_category_selector', options=populate_dropdown('categorical'),
                                             multi=True)]),
                        html.P(["Select score:",
                                dcc.Dropdown(id='expl_continuous_selector', options=populate_dropdown('continuous'))]),
                        html.P(["Select plot style:",
                                dcc.Dropdown(id='plot_selector',
                                             value=1,
                                             options=[dict(label=v, value=k) for k, v in plot_lookup.items()])]),
                        html.P("Tips:"),
                        html.P("The color of each segment indicates the mean of the selected score"),
                        html.P("The size of each segment represents the size of that student population"),
                        html.P("Click on a category to zoom in"),
                    ],
                    className="pretty_container four columns",
                ),
                html.Div([dcc.Graph(id="sunburst_plot")],
                         className="pretty_container four columns"),
                html.Div([dcc.Graph(id="second_explore_plot")],
                         className="pretty_container four columns"),
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        # ################################################< TAG3 PART >#############################################

        html.Div(
            [
                html.Div(
                    [
                        html.H1("Univariate Analysis"),
                        html.P(
                            [
                                "Select a continuous variable:",
                                dcc.Dropdown
                                    (
                                    id="continuous_selector",
                                    options=populate_dropdown('continuous'),
                                    # className="dcc_control",
                                ),
                            ]
                        ),
                        html.P(
                            [
                                "Select hist width:",
                                dcc.Slider(
                                    id="width_slider",
                                    min=2,
                                    max=20,
                                    value=5,
                                    marks={str(2): str(2), str(5): str(5), str(20): str(20)},
                                    # className="dcc_control",
                                ),
                            ]
                        ),

                    ],
                    className="pretty_container four columns",
                    id="univariate analysis",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="max_value"), html.P("Max Value")],
                                    id="wells",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="min_value"), html.P("Min Value")],
                                    id="gas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="mean_value"), html.P("Mean Value")],
                                    id="oil",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="median_value"), html.P("Median Value")],
                                    id="water",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="hist_plot")],
                            id="adjustableHistPlot",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        ######################################################< TAG4 PART >##################################################

        html.Div(
            [
                html.Div(
                    [
                        html.H1("Get your report"),
                        dbc.Button("Report generator", id="open-xl"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Report"),
                                dbc.ModalBody(
                                    html.Pre(
                                        report_text
                                    ),
                                    id="Report_body"
                                ),
                                dbc.ModalFooter([
                                    dbc.Button("Save", id="save-xl", className="ml-auto"),  # todo: save
                                    dbc.Button("Close", id="close-xl", className="ml-auto"), ]
                                ),
                            ],
                            id="modal-xl",
                            size="xl",
                            centered=True,
                        ),
                    ],
                    id="report",
                    className="pretty_container four column",
                )
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        # TODO: more graphs

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Report modal
@app.callback(
    Output("modal-xl", "is_open"),
    [Input("open-xl", "n_clicks"), Input("close-xl", "n_clicks"), Input("save-xl", "n_clicks")],
    [State("modal-xl", "is_open")],
)
def toggle_modal(n1, n2, n3, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# Four blocks above histogram
@app.callback(
    [
        Output("max_value", "children"),
        Output("min_value", "children"),
        Output("mean_value", "children"),
        Output("median_value", "children"),
    ],
    [Input('continuous_selector', 'value')],
)
def update_text(data):
    if not data:
        return "", "", "", ""
    data = get_field_data(data, file_loc=student_data_file).dropna()
    return str(max(data)), str(min(data)), str(round(np.mean(data), 2)), str(np.median(data))


# Adjustable histogram
@app.callback(Output('hist_plot', 'figure'),
              [Input('continuous_selector', 'value'), Input('width_slider', 'value')])
def make_hist_plot(fields, bar_width):
    if not fields:
        return {'data': []}
    else:
        data = get_field_data(fields, file_loc=student_data_file)
        Width = (max(data) - min(data)) / bar_width
        data = get_binned_data(fields, Width, file_loc=student_data_file)
        fig = go.Figure(data=[go.Bar(
            x=data["range"],
            y=data["count"],
            width=[Width] * bar_width,
            name="Adjustable Histogram"
        )])
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        return fig


def get_empty_sunburst(text: str):
    """
    Generates and empty sunburst plot with `text` at its center

    :param text: informational text to display
    :return: `plotly` figure
    """
    return px.sunburst(
        {'x'    : [text],
         'value': [1]},
        path=['x'],
        hover_data=None
    )


@app.callback(Output('second_explore_plot', 'figure'),
              [Input('expl_category_selector', 'value'), Input('expl_continuous_selector', 'value'),
               Input('plot_selector', 'value')])
def make_second_explore_plot(categorical: list, continuous, plot):
    """
    Make a plot based on the categorical and continuous data selected. Choose a box plot or frequency plot depending
    on the plot selected.

    :param categorical: list of data categories
    :param continuous: single continuous data field
    :param plot: "frequency plot" or "box plot"
    :return: `plotly` figure
    """
    if not categorical:
        return {'data': []}
    elif plot_lookup[plot] == 'frequency plot':
        fig = get_frequency_plot(categorical)
    elif plot_lookup[plot] == 'box plot':
        if continuous:
            fig = get_box_plot(categorical, continuous)
        else:
            fig = get_empty_sunburst("select a score")
    else:
        raise ValueError(f"{plot} is not a valid plot option")

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


def get_box_plot(categorical, continuous):
    """
    Create a box plot given the categories as the x axis and the continuous field as the y-axis

    :param categorical: list of categorical data fields
    :param continuous: single continuous variable
    :return: `plotly` figure
    """
    data = get_field_data((categorical[0], continuous), file_loc=student_data_file)
    fig = px.box(data, x=categorical[0], y=continuous)
    return fig


def get_frequency_plot(categorical):
    """
    Create a frequency plot of the count of each category

    :param categorical: list of categorical data fields
    :return: `plotly` figure
    """
    data, _ = get_hierarchical_data(categorical, file_loc=student_data_file)
    fig = px.bar(data, x=categorical[0], y='count')
    return fig


# TODO: use state callback to update instead of creating new figure [State('graph', 'figure')]
@app.callback(Output('sunburst_plot', 'figure'),
              [Input('expl_category_selector', 'value'), Input('expl_continuous_selector', 'value')])
def make_sunburst(fields, color_var):
    """
    Callback to generate the sunburst figure based on the selected categorical input fields and the desired 
    continuous variable, used to color the segments.
    
    :param color_var: The continuous variable with which to color the segments
    :param fields: Categorical data fields with which to size segments by frequency
    :return: `plotly` figure
    """
    if not fields:
        fig = get_empty_sunburst("Select a category")
    elif not color_var:
        fig = get_empty_sunburst("Select a score")
    else:
        fig = get_sunburst_plot(color_var, fields)

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


def get_sunburst_plot(color_var, fields):
    """
    Create a sunburst plot

    :param color_var: The continuous variable with which to color the segments
    :param fields: Categorical data fields with which to size segments by frequency
    :return: `plotly` figure
    """
    data, color_var_mean = get_hierarchical_data(fields, color_var, file_loc=student_data_file)
    # TODO: scale doesn't update when the color_var is changed
    fig = px.sunburst(
        data,
        path=fields,
        values='count',
        color='mean',
        hover_data=fields,  # TODO: figure out what the best hover data is
        color_continuous_scale='Portland',
        color_continuous_midpoint=color_var_mean,
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
