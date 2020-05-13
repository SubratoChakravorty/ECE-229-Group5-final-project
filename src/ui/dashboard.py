"""
Just run using `python dashboard.py`
"""
from typing import Union, List, Tuple

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

from src.univariate_methods import return_fields

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

student_data_file = '../../data/student_data.csv'
df = pd.read_csv(student_data_file)
categories = return_fields('../../data/student_data.csv')

col_options = [dict(label=x, value=x) for x in df.columns]
app.layout = html.Div(
    [
        html.H1("Explore the Data"),
        html.P("Click a category on the inner plot to filter"),
        html.Div(
            html.P(["Select categories:", dcc.Dropdown(id='category_selector', options=col_options, multi=True)]),
            style={"width": "25%", "float": "left"},
        ),
        dcc.Graph(id="graph",
                  style={"width": "75%", "display": "inline-block"},
                  animate=False),
    ]
)


@app.callback(Output("graph", "figure"), [Input('category_selector', "value")])
def make_figure(fields):
    """
    Callback to generate the sunburst figure based on the selected categorical input fields and the desired 
    continuous variable, used to color the segments.
    
    :param fields: 
    :return: 
    """
    if fields is None:
        # TODO: return text telling the user to select data

        fig = px.sunburst(
            {'x': ["Select a category"],
             'value': [1]},
            path=['x'],
            hover_data=None
        )
        return fig

    color_var = 'X1SCIEFF'

    data, color_var_mean = get_group_counts_and_means(fields, color_var)

    fig = px.sunburst(
        data,
        path=fields,
        values='count',
        color='mean',
        hover_data=fields,  # TODO: figure out what the best hover data is
        color_continuous_scale='RdBu',
        color_continuous_midpoint=color_var_mean)
    return fig


def get_group_counts_and_means(fields: Union[List, Tuple], color_var: str) -> Tuple[pd.DataFrame, float]:
    """
    Returns a `DataFrame` filtered and grouped by `fields` with columns for the count of each group and the mean of the
    continuous `color_var` for each group.

    :param fields:
    :param color_var:
    :return: DataFrame and mean of color_var
    """
    df_filtered = df[fields + [color_var]]

    gr = df_filtered.groupby(by=fields)
    data = gr.count().reset_index().rename(columns={color_var: 'count'})
    mean = gr.mean().reset_index()[color_var]
    data['mean'] = mean

    color_var_mean = df_filtered[color_var].mean()
    return data, color_var_mean


if __name__ == '__main__':
    app.run_server(debug=True)
