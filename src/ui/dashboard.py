"""
Just run using `python dashboard.py`
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

# for kai's vscode relative path issue only
# just ignore it
# import sys
# sys.path.append("/Users/wangkai/Downloads/ece229/project/ECE-229-Group5-final-project")

from src.config import student_data_file
from src.univariate_methods import return_fields, get_counts_means_data

external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# color for frontend
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# load first page markdown file
with open("../txt/Readme.md", "r") as f:
    markdown_first_page = f.read()

app = dash.Dash(__name__, external_stylesheets=external_css)

introduction_tab = dcc.Tab(
    label="Introduction",
    children=[
    html.H1(
        children="Analysis of Ninth Grader’s Science Self Efficacy",
        style={
            'textAlign': 'center',
            'color': colors['text'],
        }
    ),
    html.H2(
        children="ECE229 - Team 5",
        style={
            'textAlign': 'center',
            'color': colors['text'],
        }
    ),
    # html.Div(children='Dash: A web application framework for Python.', style={
    #     'textAlign': 'center',
    #     'color': colors['text']
    # }),

    # Todo: make it looks nicer
    html.Div([
        dcc.Markdown(
            children=markdown_first_page)
    ])
    ]
)

categories = return_fields('../../data/student_data.csv')
col_options = [dict(label=x, value=x) for x in categories]

explore_tab = dcc.Tab(
    label="Explore",
    children=[
        html.H1("Explore the Data"), html.P("Click a category on the inner plot to filter"),
        html.Div(
            [
                html.P(["Select categories:", dcc.Dropdown(id='category_selector', options=col_options, multi=True)]),
                html.P(["Select score:", dcc.Dropdown(id='color_var_selector', options=col_options)]),
            ],
            style={"width": "25%", "float": "left"}
        ),
        dcc.Graph(id="graph",
                  style={"width": "75%", "display": "inline-block"},
                  animate=False),
    ]
)

inspect_tab = dcc.Tab(
    label="Inspect",
    children=[
        html.H1("Univariate Analysis")
    ]
)

insights_tab = dcc.Tab(
    label="Insights",
    children=[
        html.H1("Multivariate Statistical Analysis")
    ]
)

app.layout = html.Div([
    dcc.Tabs(
        [
            introduction_tab,
            explore_tab,
            inspect_tab,
            insights_tab,
        ],
    )
])


@app.callback(Output('graph', 'figure'), [Input('category_selector', 'value'), Input('color_var_selector', 'value')])
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
        return fig
    elif not color_var:
        fig = px.sunburst(
            {'x'    : ["Select a score"],
             'value': [1]},
            path=['x'],
            hover_data=None
        )
        return fig

    data, color_var_mean = get_counts_means_data(fields, color_var, file_loc=student_data_file)

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


if __name__ == '__main__':
    app.run_server(debug=True)
