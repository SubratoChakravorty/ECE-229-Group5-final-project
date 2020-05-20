"""
Just run using `python dashboard.py`
"""
# Import required libraries
import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash.dependencies import Input, Output

from controls import WELL_STATUSES, WELL_TYPES
from src.config import variables_file, student_data_file
from src.univariate_methods import return_fields, get_counts_means_data, get_binned_data, get_field_data, get_var_info

def get_fields(fields: list, file_loc: str):
    df = pd.read_csv(file_loc)
    return df[fields]

# Style configuration
external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# color for frontend
colors = {
    'background': '#111111',
    'text'      : '#7FDBFF'
}

plot_lookup = {0: 'box plot',
               1: 'frequency plot'}

# Populate fields from data
categories = return_fields('../../data/student_data.csv')
col_types = {}
for cate in categories:
    col_types[cate] = cate
col_options = [dict(label=x, value=x) for x in categories]
print(col_types)

vars_df = get_var_info(variables_file)


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

        html.Div(  # Explore
            [
                html.Div(
                    [
                        html.H1("Explore the Data"),
                        html.P("Click a category on the inner plot to filter"),
                        html.P(["Select categories:",
                                dcc.Dropdown(id='category_selector', options=populate_dropdown('categorical'),
                                             multi=True)]),  # TODO: hover dropdown to get long text (new component)
                        html.P(["Select score:",
                                dcc.Dropdown(id='continuous_var_selector', options=populate_dropdown('continuous'))]),
                        html.P(["Select plot style:",
                               dcc.Dropdown(id='plot_selector',
                                            value=1,
                                            options=[dict(label=v, value=k) for k, v in plot_lookup.items()])]),
                    ],
                    className="pretty_container four columns",
                ),
                html.Div([dcc.Graph(id="sunburst_plot")],
                         className="pretty_container four columns"),
                html.Div([dcc.Graph(id="second_explore_plot")],
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
                            id="continuous_selector",
                            options=populate_dropdown('continuous'),
                            className="dcc_control",
                        ),
                        html.P("Select bar width:", className="control_label"),
                        dcc.Slider(
                            id="width_slider",
                            min=2,
                            max=20,
                            value=5,
                            marks={str(5): str(5), str(30): str(30)},
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
                        # html.Div(
                        #     [dcc.Graph(id="count_graph")],
                        #     id="countGraphContainer",
                        #     className="pretty_container",
                        # ),
                        html.Div([dcc.Graph(id="hist_plot", animate=False)],
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),

        ######################################################< TAG4 PART >##################################################

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

@app.callback(Output('hist_plot', 'figure'),
              [Input('continuous_selector', 'value'),Input('width_slider', 'value')])
def make_hist_plot(fields,bar_width):
    if not fields:
        return {'data': []}
    else:
        data = get_field_data(fields, file_loc=student_data_file)
        Width = (max(data)-min(data))/bar_width
        data = get_binned_data(fields, Width, file_loc=student_data_file)
        fig = go.Figure(data=[go.Bar(
            x=data["range"],
            y=data["count"],
            width=[Width]*bar_width # customize width here
        )])
        #fig = px.bar(data, x=fields[0], y='count')
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        return fig

def get_empty_sunburst(text:str):
    return px.sunburst(
                {'x'    : [text],
                 'value': [1]},
                path=['x'],
                hover_data=None
            )


@app.callback(Output('second_explore_plot', 'figure'),
              [Input('category_selector', 'value'), Input('continuous_var_selector', 'value'),
               Input('plot_selector', 'value')])
def make_second_explore_plot(categorical: list, continuous, plot):
    if not categorical:
        return {'data': []}
    elif plot_lookup[plot] == 'frequency plot':
        data, _ = get_counts_means_data(categorical, file_loc=student_data_file)
        fig = px.bar(data, x=categorical[0], y='count')
    elif plot_lookup[plot] == 'box plot':
        if continuous:
            data = get_fields([categorical[0], continuous], file_loc=student_data_file)
            fig = px.box(data, x=categorical[0], y=continuous)
        else:
            fig = get_empty_sunburst("select a score")
    else:
        raise ValueError(f"{plot} is not a valid plot option")
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


# TODO: use state callback to update instead of creating new figure [State('graph', 'figure')]
@app.callback(Output('sunburst_plot', 'figure'),
              [Input('category_selector', 'value'), Input('continuous_var_selector', 'value')])
def make_sunburst(fields, color_var):
    """
    Callback to generate the sunburst figure based on the selected categorical input fields and the desired 
    continuous variable, used to color the segments.
    
    :param fields: 
    :return: 
    """
    if not fields:
        fig = get_empty_sunburst("select a category")
    elif not color_var:
        fig = get_empty_sunburst("select a score")
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
