# First attempts at dash
# referencing https://dash.plotly.com/interactive-graphing, https://dash.plotly.com/basic-callbacks 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import json
import pandas as pd
import numpy as np

# load data
from data_loaders import load_data
jhs = load_data()

df = jhs.meta
data = jhs.data

# Cluster things
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(affinity='cosine', linkage='complete',
                                n_clusters=4)


available_datasets = df['dataset'].unique()
available_affinity = ['cosine', 'euclidean', 'l1', 'l2', 'manhattan']
available_linkage = [ 'complete', 'ward', 'average', 'single']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # Top row, selection items
    html.Div([
        html.Div([
            dcc.Dropdown(
                        id='dataset-filter-dropdown',
                        options=[{'label': i, 'value': i} for i in available_datasets],
                        value=available_datasets[0]), 
        ], style={'width':'48%', 'display':'inline-block'}),
        html.Div([
            dcc.Input(id='num-clusters-range', type='number',
                        min=1, max=177, step=1, 
                        value=4)
        ], style={'width':'48%', 'float':'right', 'display':'inline-block'}),

        html.Div([
            html.I('Affinity (distance metric):'),
            dcc.Dropdown(id='affinity-dropdown',
                        options=[{'label': i, 'value': i} for i in available_affinity],
                        value=available_affinity[0]),
        ], style={'width':'48%', 'display':'inline-block'}),
        html.Div([
            html.I('Linkage:'),
            dcc.Dropdown(id='linkage-dropdown',
                        options=[{'label': i, 'value': i} for i in available_linkage],
                        value=available_linkage[0])
        ], style={'width':'48%', 'float':'right', 'display':'inline-block'})
    ], style={'borderBottom':'thin lightgrey solid', 'padding':'10px 5px'}), 

    # plots 
    html.Div([
        dcc.Graph(id='graph-map', clickData={'points':[{'hovertext':1}]}),
    ], style={'width':'49%', 'display':'inline-block'}),
    html.Div([
        dcc.Graph(id='graph-1D')
    ], style={'width': '49%', 'float':'right' })

])

@app.callback(
    Output('graph-map', 'figure'),
    [Input('dataset-filter-dropdown', 'value'), 
     Input('num-clusters-range', 'value'),
     Input('affinity-dropdown', 'value'),
     Input('linkage-dropdown', 'value')])
def update_figure(selected_dataset, n_clusters, affinity, linkage):
    filtered_df = df[df.dataset == selected_dataset]

    # generate labels
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
                                n_clusters=n_clusters)
    model.fit(data[filtered_df.index])

    # fig = go.Figure()
    fig = px.scatter(filtered_df, x='py', y="px", 
                     color=model.labels_, hover_name="data_pt", 
                     color_continuous_scale=px.colors.sequential.Rainbow,
                     size_max=55, width=600, height=600)
    fig.update_traces(marker=dict(size=20, symbol='square'))
    fig.update_xaxes(autorange='reversed')
    fig.update_layout(transition_duration=500)


    return fig

@app.callback(
    Output('graph-1D', 'figure'),
    [Input('graph-map', 'clickData'), 
     Input('dataset-filter-dropdown', 'value')])
def update_1D(clickData, selected_dataset):
    # Progressively filtering dataset
    point_no = int(clickData['points'][0]['hovertext'])

    dff = df[df['dataset']==selected_dataset]
    dfff = dff[dff['data_pt']==point_no] # should be single point
    print(dfff.head())

    data_1D = data[dfff.index[0]]
    x_list = np.linspace(dfff['q_min'].iloc[0],dfff['q_max'].iloc[0],num=dfff['num_points'].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_list, y=data_1D, name=point_no))
    fig.update_layout(xaxis_title='q', yaxis_title='I', showlegend=True)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)