"""
Usage:
``python -m src``
"""
import math
import numpy as np
import pandas as pd
import plotly.express as px
from src.ui import app, cache
from itertools import product
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from typing import List, Union, Dict, Tuple
from dash.dependencies import Input, Output, State
from src.config import variables_file, student_data_file
from src.multivariate_methods import get_correlation_matrix, get_feature_importance, MLmodel
from src.univariate_methods import get_hierarchical_data, get_var_info, get_field_data, get_binned_data, get_stats, \
    get_categories

# color for frontend
colors = {
    'background': '#111111',
    'text'      : '#7FDBFF'
}

plot_lookup = {0: 'box plot',
               1: 'frequency plot'}

# Populate fields from data
vars_df = get_var_info(variables_file)
vars_df['code'] = vars_df.index
self_efficacy_predictors = ['COST_PERCEPTION', 'TCH_PRCVD_ATT', 'X1SCIID', 'X1SES', 'X1GEN']

UCSD_text = r"""
     _______.     _______.    _______       .______   ____    ____        __    __       ______         _______.    _______  
    /       |    /       |   |   ____|      |   _  \  \   \  /   /       |  |  |  |     /      |       /       |   |       \ 
   |   (----`   |   (----`   |  |__         |  |_)  |  \   \/   /        |  |  |  |    |  ,----'      |   (----`   |  .--.  |
    \   \        \   \       |   __|        |   _  <    \_    _/         |  |  |  |    |  |            \   \       |  |  |  |
.----)   |   .----)   |      |  |____       |  |_)  |     |  |           |  `--'  |    |  `----.   .----)   |      |  '--'  |
|_______/    |_______/       |_______|      |______/      |__|            \______/      \______|   |_______/       |_______/ 
                                                                                                                                

                                        """


def populate_dropdown(category: str = None) -> List[dict]:
    """
    Generate a list of dictionaries to use to populate the dropdown menus

    :param category: 'continuous' or 'categorical'. If `None` select all variables.
    :return: a list of dicts with keys 'label' and 'value'
    """
    if category is not None:
        assert category in ['continuous',
                            'categorical'], f"category must be 'continuous' or 'categorical', not {category}"
        df = vars_df.loc[vars_df['type'] == category, 'short']
    else:
        df = vars_df['short']
    return [dict(label=v, value=k) for k, v in df.to_dict().items()]


def fig_formatter(**kw):
    """
    Decorator for functions that produce figures. By default, all margins are stripped, but the margin sized can be
    set individually.

    :param t: top margin
    :param l: left margin
    :param r: right margin
    :param b: bottom margin
    :return:
    """
    t = kw.get('t', 0)
    l = kw.get('l', 0)
    r = kw.get('r', 0)
    b = kw.get('b', 0)

    def wrap(func):
        def wrapped(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.update_layout(margin=dict(t=t, l=l, r=r, b=b),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              )
            return fig

        return wrapped

    return wrap


correlation_matrix = get_correlation_matrix(vars_df.loc[vars_df['type'] == 'continuous'].index.to_list(),
                                            student_data_file, method='spearman')


@fig_formatter(t=30)
def make_correlation_heatmap() -> go.Figure:
    """
    Produce the correlation heatmap figure
    """
    short_name_lookup = vars_df.loc[correlation_matrix.columns, 'short'].to_dict()
    df = correlation_matrix.rename(columns=short_name_lookup)
    df = df.rename(index=short_name_lookup)
    fig = px.imshow(
        df,
        labels=short_name_lookup,
        x=df.index,
        y=df.columns,
    )
    fig.update_layout(autosize=True)
    return fig


def get_slider(field: str) -> html.Div:
    """
    Return a hidden div with slider text above the slider.

    :param field: A field from the list of valid fields
    :return hidden div
    """
    field_name = vars_df.loc[field, 'short']
    if vars_df.loc[field, 'type'] == 'continuous':
        minimum, median, maximum = tuple(round(v, 1) for v in get_stats(field, student_data_file))
        div = html.Div([
            html.P(children=[field_name], id=field + '_slider_state'),
            dcc.Slider(
                id=field + '_slider',
                min=minimum,
                max=maximum,
                value=median,
                step=0.1,
                updatemode='drag',
                marks={minimum: f'{minimum: .1f}',
                       median : f'{median: .1f}',
                       maximum: f'{maximum: .1f}'},
            ),
        ],
            style={'display': 'none'},
            id=field + '_slider_div',
        )
    elif vars_df.loc[field, 'type'] == 'categorical':
        mode, category_lookup = get_categories(field, student_data_file)
        div = html.Div([
            html.P(children=[field_name], id=field + '_slider_state'),
            dcc.Slider(
                id=field + '_slider',
                min=1,
                max=len(category_lookup),
                value=mode,
                step=1,
                included=False,
                updatemode='drag',
                marks={k: v[:3] if isinstance(v, str) else str(v) for k, v in category_lookup.items()}
            ),
        ],
            style={'display': 'none'},
            id=field + '_slider_div',
        )
    else:
        raise ValueError(f"field {field} is invalid")
    return div


# Create app layout
app.layout = html.Div(
    [
        # Title
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
                                html.H1(
                                    "Boosting Interest in STEM",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    """An analysis of ninth graders' feelings towards science""",
                                    style={"margin-top": "0px"}
                                ),
                                html.H6(
                                    """ECE229 - Team 5: Ian Pegg, Subrato Chakravorty, Yan Sun, Daniel You, 
                                    Heqian Lu, Kai Wang""",
                                    style={"margin-top": "0px"}
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
                        ),
                    ],
                    className="three-third column",
                    id="github-button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        # Introduction
        html.Div(
            [
                dcc.Markdown(
                    """
                    ### How to use this dashboard
                    
                    #### 1. Feature Importance
                    
                    Understand what drives an interest in science. Look at the bright yellow and dark blue squares 
                    in the correlation heatmap. Those indicate factors that are strongly correlated with each other. 
                    Explore these in more detail using the selection menus. Use this understanding to choose variables
                    that you think will be important in determining whether a student will continue in STEM. Test
                    these variables in the next section.
                    
                    #### 2. Predictor
                    
                    This is an opportunity to create a "student" and see how various factors impact their interest
                    in STEM. Choose the measurable you are looking to predict, choose the student's variables, then
                    experiment to what factors might boost an interest in STEM.
                    
                    #### 3. Explore
                    
                    Gain a better understanding of the data. This section allows you to explore some of the patterns
                    you see in greater detail, and to better understand the data behind the predictions.
                    
                    ### Dataset
                    This study employs public-use data from the 
                    [High School Longitudinal Study of 2009 (HSLS:09)](https://nces.ed.gov/surveys/hsls09/). 
                    The goal of the study is to understand the factors that lead students to choose science, technology, 
                    engineering, and mathematics courses, majors, and careers.

                    The dataset can be downloaded 
                    [here](https://nces.ed.gov/EDAT/Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip).
                    """
                )
            ],
            className="pretty_container",
            style={"margin-bottom": "25px"}
        ),

        # Correlations
        html.Div(
            [
                html.Div([
                    html.H1("Feature Importance"),
                    dcc.Markdown("""##### Correlation heatmap:\n- Hover to view details\n- Zoom by selecting an area 
                    of interest\n- Double-click to zoom out"""),
                    html.Div([dcc.Graph(id="correlation_matrix", figure=make_correlation_heatmap())], ),
                ],
                    className="pretty_container six columns"
                ),
                html.Div([
                    html.Div([
                        dcc.Markdown("""##### Continuous variables importance:\n- Large positive values have a 
                        strong positive correlation with the selected dependent variable.\n- Large negative 
                        values have a strong negative correlation with the selected dependent variable."""),
                        html.P([
                            "Select x-axis:",
                            dcc.Dropdown(id='import_x_selector', options=populate_dropdown('continuous'), multi=True,
                                         value=['N1SCIYRS912', 'X1SCIUTI', 'X3TGPAENG', 'X3TGPAMAT', 'X3TGPASCI',
                                                # continuous
                                                'S1STCHFAIR_neg', ]),
                            dcc.Dropdown(id='import_y_selector', options=populate_dropdown('continuous'),
                                         value='X1SCIEFF'),
                        ]),
                    ],
                        id="FIselector",
                        # className="mini_container"
                    ),
                    html.Div([
                        html.P([
                            html.H6("Importance bar plot for continuous variables"),
                            dcc.Graph(id="importance_bar1"),
                        ],
                        ),
                    ],
                        id="importance_bar",
                        # className="mini_container"
                    ),
                ],
                    id="FI-column",
                    className="pretty_container six columns",
                ),
            ],
            className="flex-display",
        ),
        html.Div(
            [
                html.P([
                    html.H6("Understand the correlations between categorical variables and a dependent variable"),
                    "Select categories:",
                    dcc.Dropdown(id='importance_category_x_selector', options=populate_dropdown('categorical'),
                                 multi=True,
                                 value=['COURSE_TYPE', 'N1GEN', 'N1GROUP', 'SCH_LOCALE']),
                    "Select dependent variable",
                    dcc.Dropdown(id='importance_category_y_selector', options=populate_dropdown('continuous'),
                                 value='X1SCIEFF'),
                    dcc.Markdown(
                        """Notes:\n- Left: Feature importance can be viewed as a relative scale generated using the 
                        one-way [ANOVA test](https://en.wikipedia.org/wiki/One-way_analysis_of_variance).\n- Right: 
                        p-value is the probability that categories share the same mean in the dependent variable. 
                        Categories with p-values above the dotted line can be rejected as not predictive of the 
                        dependent variable. Categories with very small p-values are strongly predictive of the 
                        dependent variable. """
                    )
                ],
                    className="pretty_container four columns"
                ),
                html.Div(dcc.Graph(id="categorical_importance_bar"),
                         className="pretty_container four columns"),
                html.Div(dcc.Graph(id="categorical_p_bar"),
                         className="pretty_container four columns"),
            ],
            className="flex-display",
        ),

        # ML Model
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Predictor"),
                        html.P(
                            [
                                "Select variables:",
                                dcc.Dropdown(
                                    id="ml_independent_var_selector",
                                    options=populate_dropdown(),
                                    value=['COST_PERCEPTION', 'TCH_PRCVD_ATT', 'X1SCIID', 'X1SCIINT', 'X1SCIUTI',
                                           'X1GEN'],
                                    multi=True
                                ),
                                "Select value to predict:",
                                dcc.Dropdown(
                                    id="ml_dependent_var_selector",
                                    options=populate_dropdown('continuous'),
                                    value='X1SCIEFF'
                                ),
                                "Select x-axis:",
                                dcc.Dropdown(
                                    id="ml_x_axis_selector",
                                    options=populate_dropdown('continuous'),
                                    value='X1SES'
                                ),
                            ]
                        ),
                        html.Div([get_slider(field) for field in vars_df.index], id='ml_sliders'),
                    ],
                    className="pretty_container four columns",
                    id="ml_controls",
                ),
                html.Div(
                    [
                        dcc.Markdown("""### Create and test your student:\n- Use the dropdowns to select the parameters 
                    you would like to adjust and the parameter you would like to predict\n- Use the sliders to adjust 
                    the values of your chosen parameters\n- This model was generated using random forest regression 
                    and is re-trained each time a new dropdown selection is made. Please allow it a few seconds to 
                    complete training after each change."""),
                        dcc.Graph(id="ml_prediction_plot")
                    ],
                    className="pretty_container eight columns",
                ),
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        # Explore
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Explore"),
                        html.H6("Investigate the relationships between categorical variables"),
                        html.P("Click a category on the inner plot to filter"),
                        html.P(["Select categories:",
                                dcc.Dropdown(id='expl_category_selector', options=populate_dropdown('categorical'),
                                             multi=True, value=['COURSE_TYPE', 'N1HIDEG', 'SCIJOB'])]),
                        html.P(["Select score:",
                                dcc.Dropdown(id='expl_continuous_selector', options=populate_dropdown('continuous'),
                                             value='X1SCIEFF'), ]),
                        html.P(["Select plot style:",
                                dcc.Dropdown(id='plot_selector',
                                             value=1,
                                             options=[dict(label=v, value=k) for k, v in plot_lookup.items()])]),
                    ],
                    className="pretty_container four columns",
                    id="explore-part"
                ),
                html.Div([dcc.Graph(id="sunburst_plot"),
                          html.P("Tips:"),
                          html.P("The color of each segment indicates the mean of the selected score"),
                          html.P("The size of each segment represents the size of that student population"),
                          html.P("Click on a category to zoom in"), ],
                         className="pretty_container five columns",
                         id="sunburst-div"),
                html.Div([dcc.Graph(id="second_explore_plot"),
                          html.P("Tips:"),
                          html.P("The x-axis is the first-selected categorical variable"), ],
                         className="pretty_container five columns",
                         id="sunburst-bar-chart"),
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        # Histogram
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown("""##### Histogram for continuous variables:\n- Use the dropdown to select the 
                        continuous numerical variable of interest.\n- Use the slider to adjust the histogram 
                        resolution."""),
                        html.P(
                            [
                                "Select a continuous variable:",
                                dcc.Dropdown
                                    (
                                    id="continuous_selector",
                                    options=populate_dropdown('continuous'),
                                    value='X1SCIEFF'
                                ),
                            ]
                        ),
                        html.P(
                            [
                                "Select bin width:",
                                dcc.Slider(
                                    id="width_slider",
                                    min=2,
                                    max=20,
                                    value=10,
                                    marks={'2': '2', '5': '5', '10': '10', '20': '20'},
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

        # Variable reference table
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Variables Reference"),
                        dbc.Table.from_dataframe(vars_df, striped=True, bordered=True, hover=True)
                    ],
                    id='var-table',
                    className="pretty_container four column",
                )
            ],
            className="flex-display",
            style={"margin-bottom": "25px"}
        ),

        # Report
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
                                    [
                                        html.Pre(UCSD_text, id="Report_body"),
                                        html.H2("This is your profile"),
                                        html.Pre(id="report_text"),
                                        dcc.Graph(id="ml_prediction_plot2")
                                    ]
                                ),
                                dbc.ModalFooter([
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

        # Documentation link
        html.Div(
            [
                html.A(
                    html.Button("documentation", id="documentation-button"),
                    href="http://ecetestdoc.com.s3-website-us-west-2.amazonaws.com",
                ),
            ],
            className="two-half column",
            id="doc-button",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


@app.callback(
    [Output('categorical_importance_bar', 'figure'),
     Output('categorical_p_bar', 'figure')],
    [Input('importance_category_x_selector', 'value'),
     Input('importance_category_y_selector', 'value')]
)
def make_categorical_importance_plots(exog: List[str], endog: str) -> Tuple[go.Figure, go.Figure]:
    """
    Callback to generate the categorical feature importance plots

    :param exog: exogenous (independent) variables
    :param endog: endogenous (dependent) variable
    :return: tuple of figures: (importance bar plot, p-value plot)
    """
    if not exog:
        fig1 = get_empty_sunburst("Select a category")
        fig2 = fig1
    elif not endog:
        fig1 = get_empty_sunburst("Select a dependent variable")
        fig2 = fig1
    else:
        fi_dict = get_feature_importance(endog, exog)['categorical']
        fi_dict = {vars_df.loc[k, 'short']: v for k, v in fi_dict.items()}
        fig1 = get_categorical_importance_plot(fi_dict)
        fig2 = get_categorical_p_plot(fi_dict)
    return fig1, fig2


@fig_formatter(t=25)
def get_categorical_importance_plot(fi_dict: Dict[str, Tuple[int, int]]) -> go.Figure:
    """
    Generate the categorical importance bar plot given the feature importance dictionary

    :param fi_dict: feature importance dictionary. Independent variables are keys, values are tuple(importance, p-value)
    :return: Bar plot figure
    """
    importance_dict = {k: v[0] for k, v in fi_dict.items()}
    return go.Figure(go.Bar(
        x=list(importance_dict.keys()),
        y=list(importance_dict.values()),
        marker={'color': list(importance_dict.values())},
    ),
        layout=dict(title="Categorical Feature Importance")
    )


@fig_formatter(t=25)
def get_categorical_p_plot(fi_dict: Dict[str, Tuple[int, int]]) -> go.Figure:
    """
    Generate the p-value bar plot given the feature importance dictionary

    :param fi_dict: feature importance dictionary. Independent variables are keys, values are tuple(importance, p-value)
    :return: Bar plot figure
    """
    p_dict = {k: v[1] for k, v in fi_dict.items()}
    return go.Figure([
        go.Bar(
            x=list(p_dict.keys()),
            y=list(p_dict.values()),
            marker={'color': list(p_dict.values())},
        ),
        go.Scatter(
            x=[list(p_dict.keys())[0], list(p_dict.keys())[-1]],
            y=[0.05, 0.05],
            mode='lines',
            line=dict(dash='dash'),
        ),
    ],
        layout=dict(
            title="Statistical Significance (p-value)",
            showlegend=False,
            yaxis=dict(type="log"),
        ),
    )


# Report modal
@app.callback(
    Output("modal-xl", "is_open"),
    [Input("open-xl", "n_clicks"), Input("close-xl", "n_clicks")],
    [State("modal-xl", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    """
    Modal style report window

    :param n1: click or not for open button
    :param n2: click or not for close button
    :param is_open: boolean 
    :return: boolean is_open
    """
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
    """
    Update 4 small windows above histogram

    :param data: column data in selected field
    :return: 4 statistical number, max, min, mean, median number
    """
    if not data:
        return "", "", "", ""
    data = get_field_data(data, file_loc=student_data_file).dropna()
    return str(max(data)), str(min(data)), str(round(np.mean(data), 2)), str(np.median(data))


# Adjustable histogram
@app.callback(Output('hist_plot', 'figure'),
              [Input('continuous_selector', 'value'), Input('width_slider', 'value')])
def make_hist_plot(fields, bar_width):
    """
    Histogram plot callback

    :param fields: content of dropdown menu
    :param bar_width: value of slider
    :return: `plotly` figure
    """
    if not fields:
        return {'data': []}
    else:
        fig = get_histogram(bar_width, fields)
        return fig


@fig_formatter()
def get_histogram(bar_width, fields):
    """
    Generate a histogram plot

    :param bar_width: The histogram bin width
    :param fields: The continuous variable to examine
    :return: `plotly` histogram figure
    """
    data = get_field_data(fields, file_loc=student_data_file)
    width = (max(data) - min(data)) / bar_width
    data = get_binned_data(fields, width, file_loc=student_data_file)
    fig = go.Figure(data=[go.Bar(
        x=data["range"],
        y=data["count"],
        width=[width] * bar_width,
        name="Adjustable Histogram"
    )])
    return fig


@fig_formatter()
def get_empty_sunburst(text: str):
    """
    Generates an empty sunburst plot with `text` at its center

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
    return fig


@fig_formatter()
def get_box_plot(categorical, continuous):
    """
    Create a box plot given the categories as the x axis and the continuous field as the y-axis

    :param categorical: list of categorical data fields
    :param continuous: single continuous variable
    :return: `plotly` figure
    """
    labels = vars_df.loc[categorical + [continuous], 'short'].to_dict()
    data = get_field_data((categorical[0], continuous), file_loc=student_data_file)
    fig = px.box(data, x=categorical[0], y=continuous, labels=labels)
    return fig


@fig_formatter()
def get_frequency_plot(categorical):
    """
    Create a frequency plot of the count of each category

    :param categorical: list of categorical data fields
    :return: `plotly` figure
    """
    labels = vars_df.loc[categorical, 'short'].to_dict()
    data, _ = get_hierarchical_data(categorical, file_loc=student_data_file)

    # Generate stacked bars if there is more than one category
    if len(labels) > 1:
        it = product(*tuple(data[c].unique() for c in categorical[1:]))
        x = data[categorical[0]].unique()
        bars = []
        for i in it:
            df_filter = [all(z) for z in zip(*tuple(data[c] == v for c, v in zip(categorical[1:], i)))]
            b = go.Bar(
                name='/'.join(i),
                x=x,
                y=data.loc[df_filter, 'count']
            )
            bars.append(b)
        fig = go.Figure(bars, layout=dict(barmode='stack'))
    else:
        fig = px.bar(data, x=categorical[0], y='count', labels=labels)
    return fig


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

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


@fig_formatter()
def get_sunburst_plot(color_var, fields):
    """
    Create a sunburst plot

    :param color_var: The continuous variable with which to color the segments
    :param fields: Categorical data fields with which to size segments by frequency
    :return: `plotly` figure
    """
    data, color_var_mean = get_hierarchical_data(fields, color_var, file_loc=student_data_file)
    fig = px.sunburst(
        data,
        path=fields,
        values='count',
        color='mean',
        color_continuous_scale='balance',
        color_continuous_midpoint=color_var_mean,
    )
    return fig


@app.callback(Output('importance_bar1', 'figure'),
              [Input('import_x_selector', 'value'), Input('import_y_selector', 'value')])
def make_importance_bar_plot(exog: List[str], endog: str):
    """
    Create the importance bar plot

    :param exog: exogenous (independent) variables list
    :param endog: endogenous (dependent) variable, typically self-efficacy
    :return: `plotly` figure
    """
    if not exog:
        fig = get_empty_sunburst("Select an x variable")
    elif not endog:
        fig = get_empty_sunburst("Select a y variable")
    else:
        fig = get_importance_bar_plot(exog, endog, "continuous")
    return fig


@fig_formatter()
def get_importance_bar_plot(exog: List[str], endog: str, var_type: str):
    """
    Create the importance bar plot for continuous variables and p value bar for categorical variables.

    :param exog: exogenous (independent) variables
    :param endog: endogenous (dependent) variable, typically self-efficacy
    :param var_type: the type of x variables, "continuous" or "categorical"
    :return: `plotly` figure
    """
    assert isinstance(exog, list), f"The exog must be a list, not {type(exog)}"
    assert isinstance(endog, str), f"The endog must be a string, not {type(endog)}"
    assert isinstance(var_type, str), f"The var_type must be a string, not {type(var_type)}"
    for item in exog:
        assert isinstance(item, str), f"elements of exog must be strings, not {type(item)}"
    importance_dict = get_feature_importance(endog, fields=exog)
    importance = []
    fields = []
    for field in exog:
        if var_type == "continuous" and vars_df.loc[field]['type'] == var_type:
            importance.append(importance_dict[var_type][field])
            fields.append(field)
        elif var_type == "categorical" and vars_df.loc[field]['type'] == var_type:
            importance.append(math.log(importance_dict[var_type][field][0]))
            fields.append(field)
        else:
            continue
    series = pd.Series(importance, index=fields, name=endog)
    short_name_lookup = vars_df.loc[correlation_matrix.columns, 'short'].to_dict()
    series = series.rename(index=short_name_lookup)
    fig = px.bar(
        series,
        x=series.index,
        y=endog,
        color=endog,
        labels=dict(x='')
    )
    return fig


@app.callback(Output('ml_sliders', 'children'),
              [Input('ml_independent_var_selector', 'value')],
              [State('ml_sliders', 'children')],
              prevent_initial_call=False)
def show_ml_sliders(fields: List, state: List):
    """
    Show the sliders that were selected using the multiple dropdown. Hide the others.

    :param fields: List of fields
    :param state: children of the ml_sliders <P>
    :return: updated state
    """
    for n, f in enumerate(vars_df.index):
        if f in fields:
            state[n]['props']['style'] = None
        else:
            state[n]['props']['style'] = dict(display='none')
    return state


def assign_slider_text_update_callback(field: str) -> None:
    """
    Register a callback on the text above categorical sliders. It will then update that text according to the current
    selection.

    :param field: the categorical data field
    """

    if vars_df.loc[field, 'type'] == 'categorical':
        _, category_lookup = get_categories(field, student_data_file)

        def slider_text_update(value: int):
            return [f"{vars_df.loc[field, 'short']} | {category_lookup[value]}"]
    else:
        def slider_text_update(value: float):
            return [f"{vars_df.loc[field, 'short']} | {value:.1f}"]

    app.callback(output=Output(field + '_slider_state', 'children'),
                 inputs=[Input(field + '_slider', 'value')],
                 prevent_initial_call=False)(slider_text_update)


for field in vars_df.index:
    assign_slider_text_update_callback(field)

slider_inputs = [Input(field + '_slider', 'value') for field in vars_df.index]


@app.callback([Output('ml_prediction_plot', 'figure'), Output('ml_prediction_plot2', 'figure')],
              [Input('ml_independent_var_selector', 'value'),
               Input('ml_dependent_var_selector', 'value'),
               Input('ml_x_axis_selector', 'value')] + slider_inputs)
def make_prediction_plot(exog: List, endog: str, x_var: str, *slider_values: Tuple[float, ...]):
    """
    Callback to generate the prediction plot

    :param exog: exogenous (independent) variable names as a list
    :param endog: endogenous (dependent) variable name
    :param x_var: another exogenous variable to be used as the plot's x-axis
    :param slider_values: tuple of the values of the sliders, including the hidden ones
    :return: plotly figure
    """
    if not (exog and endog and x_var):
        if not exog:
            return [get_empty_sunburst("Select variables")] * 2
        elif not endog:
            return [get_empty_sunburst("Select value to predict")] * 2
        elif not x_var:
            return [get_empty_sunburst("Select x-variable")] * 2

    n_points = 20

    # train model
    model = train_model(endog, exog, x_var)

    # create x_var range
    x_min, _, x_max = get_stats(x_var)
    x_range = np.linspace(x_min, x_max, n_points)

    # create input data
    input_data = generate_model_input(x_range, exog, slider_values, x_var, n_points)

    # predict
    y = model.predict_model(input_data)
    plt = get_line_plot(x_range, y, x_var, endog)
    return [plt] * 2


def generate_model_input(x_range: np.ndarray, exog: List[str], x_values: Tuple[float], x_var: str, n_points: int) -> \
        Dict[str, Union[np.ndarray, List]]:
    """
    Produce a dictionary to be passed to the model for prediction

    :param x_range: The values of x_var
    :param exog: The field to predict
    :param x_values: The slider values
    :param x_var: The continuous exogenous variable
    :param n_points: The number of points of the exogenous variable to use
    :return: {field: [v1, ..., vn]}
    """
    value_dict = {x: x_values[n] for n, x in enumerate(vars_df.index) if x in exog}
    value_dict = convert_category_number_to_str(value_dict)
    exog, scalar_values = tuple(zip(*[(k, v) for k, v in value_dict.items()]))
    scalar_values = np.tile(np.array(scalar_values), (n_points, 1)).T
    input_data = dict(zip(exog, scalar_values))
    input_data[x_var] = x_range
    return input_data


def convert_category_number_to_str(d: dict):
    """
    Given a dictionary of the form {field: value} convert values to their string representations if the field is
    categorical

    :param d: {field: value}
    :return: dictionary of the same form as the input
    """
    return {field: get_categories(field)[1][v] if field in vars_df.loc[vars_df['type'] == 'categorical'].index else v
            for field, v in d.items()}


@fig_formatter()
def get_line_plot(x: Union[np.ndarray, list], y: Union[np.ndarray, list], x_var: str, endog: str):
    """
    Generate a line plot

    :param x: numpy array or list of x-values
    :param y: numpy array or list of y-values
    :param x_var: name of x-variable
    :param endog: endogenous (dependent) variable name
    :return: plotly figure
    """
    y_min, _, y_max = get_stats(endog)
    return go.Figure(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(smoothing=.5,
                  shape='spline'),
    ),
        layout=dict(xaxis_title=vars_df.loc[x_var, 'short'],
                    yaxis_title=vars_df.loc[endog, 'short'],
                    yaxis_range=[y_min, y_max],
                    autosize=True, ),
    )


def add_frame(text):
    """
    Add frame to raw text.

    :param text: raw report text
    :return: report text with frame
    """
    raw_text = text.split("*")
    framed_text = ""
    width, hight = max(map(lambda x: len(x), raw_text)), len(raw_text)
    framed_text += "-" * (width + 2)
    framed_text += "\n"
    for t in raw_text:
        if not t:
            continue
        framed_text += "|"
        framed_text += t
        framed_text += " " * (width - len(t))
        framed_text += "|\n"
    framed_text += "-" * (width + 2)
    framed_text += "\n"
    return framed_text


@app.callback(Output('report_text', 'children'),
              [Input('ml_independent_var_selector', 'value'),
               Input('ml_dependent_var_selector', 'value'),
               Input('ml_x_axis_selector', 'value')] + slider_inputs)
def make_report(exog: List, endog: str, x_var: str, *slider_values: float):
    """
    Generate report text.

    :param exog: List of fields
    :param endog: y variable
    :param x_var: variable of x-axis
    :param slider_values: values of all sliders in exog fields
    :return: report text
    """
    if not exog or not endog or not x_var:
        return "Please complete the dropdown in Predictor first"
    n_points = 20
    report = ""

    # create x_var range
    x_min, _, x_max = get_stats(x_var)
    x_range = np.linspace(x_min, x_max, n_points)

    # create input data
    input_data = generate_model_input(x_range, exog, slider_values, x_var, n_points)
    look_up = vars_df['short'].to_dict()
    del input_data[x_var]
    for key in input_data:
        report += "*"
        report = report + str(look_up[key]) + ": " + str(input_data[key][0])
    report = add_frame(report)
    return report


@cache.memoize()
def train_model(endog: str, exog: List[str], x_var: str):
    """
    Train predictive model based on the chosen variables

    :param endog: endogenous (dependent) variable name
    :param exog: exogenous (independent) variable names as a list
    :param x_var: another exogenous variable that will be used as the x-axis in the plot
    :return:
    """
    model = MLmodel(student_data_file)
    fields = set(exog)
    fields.add(x_var)
    model.train_model(y=endog, fields=list(fields))
    return model


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
