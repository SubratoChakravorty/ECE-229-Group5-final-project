"""
Just run using `python dashboard.py`
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

from src.univariate_methods import return_fields, get_counts

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

student_data_file = '../../data/student_data.csv'
df = pd.read_csv(student_data_file)
fields = return_fields('../../data/student_data.csv')

col_options = [dict(label=x, value=x) for x in df.columns]
dimensions = ["x", "y"]
app.layout = html.Div(
    [
        html.H1("Explore the Data"),
        html.Div(
            [html.P([d + ":", dcc.Dropdown(id=d, options=col_options)]) for d in dimensions],
            style={"width": "25%", "float": "left"},
        ),
        dcc.Graph(id="graph",
                  style={"width": "75%", "display": "inline-block"},
                  animate=False),
    ]
)


@app.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
def make_figure(x, y):
    if x is None or y is None:
        x, y = 'X1RACE', 'X1SEX'

    path = [x, y]
    color_var = 'X1SCIEFF'

    df_filtered = df[path + [color_var]]
    gr = df_filtered.groupby(by=[x, y])
    data = gr.count().reset_index().rename(columns={color_var: 'count'})
    mean = gr.mean().reset_index()[color_var]
    data['mean'] = mean

    fig = px.sunburst(
        data,
        path=path,
        values='count',
        color='mean',
        hover_data=[x, y],
        color_continuous_scale='RdBu',
        color_continuous_midpoint=np.average(df_filtered[color_var])
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
