# First attempts at dash
# referencing https://dash.plotly.com/interactive-graphing, https://dash.plotly.com/basic-callbacks 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.express as px
import plotly.graph_objects as go

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
app.title = 'HiTp dataset explorer'

app.layout = html.Div([
    dcc.Markdown('''
    ### HiTp dataset explorer

    Quick tool using sklearn's 
    [agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
     module on a few HiTp datasets.  

    '''),

    # Top row, selection items
    html.Div([
        html.Div([
            html.B('Dataset:'),
            dcc.Dropdown(
                        id='dataset-filter-dropdown',
                        options=[{'label': i, 'value': i} for i in available_datasets],
                        value=available_datasets[0]), 
        ], style={'width':'48%', 'display':'inline-block'}),
        html.Div([
            html.B('Number of clusters:'),
            dcc.Input(id='num-clusters-range', type='number',
                        min=1, max=177, step=1, 
                        value=4)
        ], style={'width':'48%', 'float':'right', 'display':'inline-block'}),

        html.Div([
            html.B('Affinity (distance metric):   '),
            dcc.Dropdown(id='affinity-dropdown',
                        options=[{'label': i, 'value': i} for i in available_affinity],
                        value=available_affinity[0]),
        ], style={'width':'48%', 'display':'inline-block'}),
        html.Div([
            html.B('Linkage:'),
            dcc.Dropdown(id='linkage-dropdown',
                        options=[{'label': i, 'value': i} for i in available_linkage],
                        value=available_linkage[0])
        ], style={'width':'48%', 'float':'right', 'display':'inline-block'})
    ], style={'borderBottom':'thin lightgrey solid', 'padding':'10px 5px'}), 

    # plots 
    html.Div([
        dcc.Graph(id='graph-map', clickData={'points':[{'hovertext':1}]}),
    ], style={'width':'39%', 'display':'inline-block'}),
    html.Div([
        dcc.RadioItems(id='linlog-select',
                       options=[{'label': 'linear', 'value': 'linear'},
                                {'label': 'log', 'value': 'log'}], 
                       value='linear', labelStyle={'display': 'inline-block'}),
        dcc.Graph(id='graph-1D'), 
        dcc.Graph(id='graph-mean')
    ], style={'width': '59%', 'float':'right' }),

    # Intermediate value mean data
    dcc.Store(id='mean-data')
])

@app.callback(
    [Output('graph-map', 'figure'),
     Output('graph-mean', 'figure'),
     Output('mean-data', 'data')], 
    [Input('dataset-filter-dropdown', 'value'), 
     Input('num-clusters-range', 'value'),
     Input('affinity-dropdown', 'value'),
     Input('linkage-dropdown', 'value'),
     Input('linlog-select', 'value')])
def update_figure(selected_dataset, n_clusters, affinity, linkage, linlog):
    filtered_df = df[df.dataset == selected_dataset]

    # generate labels
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
                                n_clusters=n_clusters)
    model.fit(data[filtered_df.index])

    # fig = go.Figure()
    fig = px.scatter(filtered_df, x='py', y="px", 
                     color=model.labels_, hover_name="data_pt", 
                     color_continuous_scale=px.colors.sequential.Rainbow,
                     size_max=55, height=500, width=500)
    fig.update_traces(marker=dict(size=20, symbol='square'))
    fig.update_xaxes(autorange='reversed')
    fig.update_layout(transition_duration=500)
    fig.update(layout_coloraxis_showscale=False)

    class_data = {}
    class_data['labels'] = model.labels_

    # Mean figure
    fig_mean = go.Figure()
    x_list = np.linspace(filtered_df['q_min'].iloc[0], filtered_df['q_max'].iloc[0], 
                         num=filtered_df['num_points'].iloc[0])
    mean_data = {}
    for i in range(model.n_clusters_):
        cluster_data = data[filtered_df[model.labels_ == i].index]
        mean_1D = np.mean(cluster_data, axis=0)
        mean_data[i] = mean_1D

        fig_mean.add_trace(go.Scatter(x=x_list, y=mean_1D + i*np.mean(mean_1D), name=f'cluster #{i}'))
        fig_mean.update_layout(xaxis_title='q', yaxis_title='I', showlegend=True)

    fig_mean.update_yaxes(type=linlog)
    class_data['mean'] = mean_data

    return fig, fig_mean, class_data

@app.callback(
    Output('graph-1D', 'figure'),
    [Input('graph-map', 'clickData'), 
     Input('dataset-filter-dropdown', 'value'),
     Input('mean-data', 'data'),
     Input('linlog-select', 'value')])
def update_1D(clickData, selected_dataset, class_data, linlog):
    # Progressively filtering dataset
    point_no = int(clickData['points'][0]['hovertext'])

    dff = df[df['dataset']==selected_dataset]
    dfff = dff[dff['data_pt']==point_no] # should be single point
    print(dfff.head())

    data_1D = data[dfff.index[0]]
    x_list = np.linspace(dfff['q_min'].iloc[0],dfff['q_max'].iloc[0],num=dfff['num_points'].iloc[0])

    # Start building figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_list, y=data_1D, name=f'point #: {point_no}',
                             line=dict(color='black')))

    label = class_data['labels'][point_no]
    cluster_mean = class_data['mean'][str(label)]

    diff = data_1D - cluster_mean
    fig.add_trace(go.Scatter(x=x_list, y=diff, name=f'data - mean(cluster {label})', 
                             line=dict(dash='dot', color='red')))
    fig.add_trace(go.Scatter(x=x_list, y=cluster_mean, name=f'mean(cluster {label})', 
                             line=dict(color='red'), opacity=0.5))




    fig.update_layout(xaxis_title='q', yaxis_title='I', showlegend=True)
    fig.update_yaxes(type=linlog)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)