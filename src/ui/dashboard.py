"""
Just run using `python dashboard.py`
"""
# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
# import dash_table as dt

# Multi-dropdown options
from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

from src.config import variables_file, student_data_file
from src.univariate_methods import return_fields, get_counts_means_data


# Style configuration
external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# color for frontend
colors = {
    'background': '#111111',
    'text'      : '#7FDBFF'
}

# Populate fields from data
categories = return_fields('../../data/student_data.csv')
col_types = {}
for cate in categories:
    col_types[cate] = cate
col_options = [dict(label=x, value=x) for x in categories]
print(col_types)

vars_df = pd.read_csv(variables_file, index_col=0)


def populate_dropdown(category: str):
    assert category in ['continuous', 'categorical'], f"category must be 'continuous' or 'categorical', not {category}"
    df = vars_df.loc[vars_df['type'] == category, 'short']
    return [dict(label=v, value=k) for k, v in df.to_dict().items()]


well_status_options = [
    {"label": str(col_types[well_status]), "value": str(well_status)}
    for well_status in col_types
]

well_type_options = [
    {"label": str(col_types[well_type]), "value": str(well_type)}
    for well_type in col_types
]

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Overview",
)

introduction_tab = dcc.Tab(
    label='About',
    value='what-is',
    children=html.Div(className='control-tab', children=[

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

    ])
)

inspect_tab = html.Div([
    dcc.Tab(
        label="Inspect",
        children=[
            html.H1("Univariate Analysis"),
            html.P("Click a category on the inner plot to filter"),
            html.Div(
                [
                    html.P(["Select continous_var:",
                            dcc.Dropdown(id='continous_var_selector', options=col_options, multi=True)]),
                ],
                style={"width": "25%", "float": "left"}
            ),
            dcc.Graph(id="graph2",
                      style={"width": "75%", "display": "inline-block"},
                      animate=False)
        ]
    ),
    dcc.Slider(
        id='continous-slider',
        min=0,
        max=100,
        value=0,
        # marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    ),
])

insights_tab = dcc.Tab(
    label="Insights",
    children=[
        html.H1("Multivariate Statistical Analysis")
    ]
)

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
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("github", id="learn-more-button"),
                            href="https://github.com/SubratoChakravorty/ECE-229-Group5-final-project",
                        )
                    ],
                    className="one-third column",
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
        ),

        # ##############################################< TAG2 PART >############################################

        html.Div(
            [
                html.Div(
                    [
                        html.H1("Explore the Data"),
                        html.P("Click a category on the inner plot to filter"),
                        html.P(["Select categories:",
                                dcc.Dropdown(id='category_selector', options=populate_dropdown('categorical'),
                                             multi=True)]),  # TODO: hover dropdown to get long text (new component)
                        html.P(["Select score:",
                                dcc.Dropdown(id='color_var_selector', options=populate_dropdown('continuous'))]),
                    ],
                    className="pretty_container four columns",
                ),
                html.Div([dcc.Graph(id="sunburst_plot", animate=False)],
                         className="pretty_container four columns"),
                html.Div([dcc.Graph(id="frequency_plot", animate=False)],
                         className="pretty_container four columns"),
            ],
            className="row flex-display",
        ),

        # ################################################< TAG3 PART >#############################################

        html.Div(
            [
                html.Div(
                    [
                        html.H1("Univariate Analysis"),
                        html.P("Select categories:", className="control_label"),
                        dcc.Dropdown(
                            id="well_statuses",
                            options=well_status_options,
                            multi=True,
                            value=list(WELL_STATUSES.keys()),
                            className="dcc_control",
                        ),
                        html.P("Select score:", className="control_label"),
                        dcc.Dropdown(
                            id="well_types",
                            options=well_type_options,
                            multi=True,
                            value=list(WELL_TYPES.keys()),
                            className="dcc_control",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=1960,
                            max=2017,
                            value=[1990, 2010],
                            marks={str(1990): str(1990), str(2010): str(2010)},
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="well_text"), html.P("No. of Wells")],
                                    id="wells",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="gasText"), html.P("Gas")],
                                    id="gas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="oilText"), html.P("Oil")],
                                    id="oil",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="waterText"), html.P("Water")],
                                    id="water",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="count_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),

        ######################################################< TAG4 PART >##################################################

        html.Div(
            [
                insights_tab,
            ],
        ),

        # TODO: more graphs

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


@app.callback(Output('frequency_plot', 'figure'),
              [Input('category_selector', 'value')])
def make_frequency_plot(fields):
    if not fields:
        return {'data': []}
    else:
        data, _ = get_counts_means_data(fields, file_loc=student_data_file)
        fig = px.bar(data, x=fields[0], y='count')
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        return fig


# TODO: use state callback to update instead of creating new figure [State('graph', 'figure')]
@app.callback(Output('sunburst_plot', 'figure'),
              [Input('category_selector', 'value'), Input('color_var_selector', 'value')])
def make_sunburst(fields, color_var):
    """
    Callback to generate the sunburst figure based on the selected categorical input fields and the desired 
    continuous variable, used to color the segments.
    
    :param fields: 
    :return: 
    """
    if not fields:
        fig = px.sunburst(
            {'x'    : ["Select a category"],
             'value': [1]},
            path=['x'],
            hover_data=None
        )
    elif not color_var:
        fig = px.sunburst(
            {'x'    : ["Select a score"],
             'value': [1]},
            path=['x'],
            hover_data=None
        )
    else:
        data, color_var_mean = get_counts_means_data(fields, color_var, file_loc=student_data_file)

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

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


def filter_dataframe(df, well_statuses, well_types, year_slider):
    df = df[
        df["Well_Status"].isin(well_statuses)
        & df["Well_Type"].isin(well_types)
        & (df["Date_Well_Completed"] > dt.datetime(year_slider[0], 1, 1))
        & (df["Date_Well_Completed"] < dt.datetime(year_slider[1], 1, 1))
        ]
    return df


if __name__ == '__main__':
    app.run_server(debug=True)
