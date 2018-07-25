#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility functions """


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools

warnings.simplefilter('ignore')


def categorical_plot(data, feature, width=800, height=400):
    """ function to plot the categorical variable """
    # make subplots
    titles = ('Distribution Plot', 'Default Rate Distribution')
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)
    
    # fig 1: count distribution for each feature
    grouped = data.groupby('target')[feature]
    values = grouped.apply(lambda x: x.value_counts(normalize=True)).unstack()
    names = list(values.columns)
    x = ['status 0', 'status 1']
    for name in names:
        trace = go.Bar(x=x, y=list(values[name]), name=name)
        fig.append_trace(trace, 1, 1)

    # fig 2: default rate bar plot for each feature
    means = data.groupby(feature)['target'].mean()
    stds = data.groupby(feature)['target'].std()
    for name, mean, std in zip(names, means[names], stds[names]):
        low, high = stats.norm.interval(0.05, loc=mean, scale=std)
        er = mean - low
        trace = go.Bar(x=[name], y=[mean], error_y=dict(array=[er], visible=True), 
                       name=name, xaxis='x2')
        fig.append_trace(trace, 1, 2)
    
    # layout setting
    legend = dict(orientation='h', xanchor='auto', y=-0.2)
    margin=go.layout.Margin(l=50, r=50, b=50, t=40, pad=4)
    fig['layout'].update(xaxis=dict(domain=[0, 0.47]), xaxis2=dict(domain=[0.53, 1]),
                         yaxis2=dict(anchor='x2'), width=width, height=height, 
                         margin=margin, legend=legend)
    fig['layout']['xaxis1'].update(title='Loan Status')
    fig['layout']['yaxis1'].update(title='Probability Density')
    fig['layout']['xaxis2'].update(title=feature.capitalize())
    fig['layout']['yaxis2'].update(title='Default Rate')

    return fig


def numerical_plot(data, feature, width=800, height=400, bins=50):
    """ function to plot the numerical variable """
    # make subplots
    titles = ('Histogram Plot', 'Default Rate vs. ' + feature.capitalize())
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)
    
    # fig 1: histogram for different loan status
    x0 = data[data['target']==0][feature]
    x1 = data[data['target']==1][feature]
    
    # find the minimum and maximum values
    start = min(x0.min(), x1.min())
    end = max(x0.max(), x1.max())
    n_unique = len(data[feature].unique())
    if n_unique <= min(end - start + 1, bins):
        bin_size = 1
    else:
        bin_size = (end - start) / bins

    # Group data together
    hist_data = [x0, x1]
    group_labels = ['Status 0', 'Status 1']

    # Create distplot
    fig1 = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, 
                              bin_size=bin_size, show_rug=False)
    displot = fig1['data']
    
    # add histgram into the final figure
    fig.append_trace(displot[0], 1, 1)
    fig.append_trace(displot[1], 1, 1)
    fig.append_trace(displot[2], 1, 1)
    fig.append_trace(displot[3], 1, 1)
    
    # fig 2: default rate bar plot for each feature or scatter plot
    if n_unique <= bins:
        # default rate bar plot
        means = data.groupby(feature)['target'].mean()
        stds = data.groupby(feature)['target'].std()
        names = list(means.index)
        for name, mean, std in zip(names, means[names], stds[names]):
            low, high = stats.norm.interval(0.05, loc=mean, scale=std)
            er = mean - low
            trace = go.Bar(x=[name], y=[mean], error_y=dict(array=[er], visible=True), 
                           name=name, xaxis='x2')
            fig.append_trace(trace, 1, 2)
    else:
        # scatter plot 
        mean = train.groupby(feature)['target'].mean()
        sem = train.groupby(feature)['target'].sem().fillna(value=0)
        index = mean.index

        lower = go.Scatter(x=index, y=mean[index]-sem[index], mode='lines', 
                           marker=dict(color="#444"), line=dict(width=0), 
                           showlegend=False)
        
        trace = go.Scatter(name='Default Rate', x=index, y=mean[index],  
                           line=dict(color='rgb(31, 119, 180)', width=1), 
                           fillcolor='rgba(68, 68, 68, 0.3)', mode='lines',)
        
        upper = go.Scatter(x=index, y=mean[index]+sem[index], mode='lines', 
                           marker=dict(color="#444"), line=dict(width=0), 
                           fill='tonexty', fillcolor='rgba(68, 68, 68, 0.3)', 
                           showlegend=False)

        fig.append_trace(lower, 1, 2)
        fig.append_trace(upper, 1, 2)
        fig.append_trace(trace, 1, 2)

    # layout setting
    legend = dict(orientation='h', xanchor='auto', y=-0.2)
    margin=go.layout.Margin(l=50, r=50, b=50, t=40, pad=4)
    fig['layout'].update(xaxis=dict(domain=[0, 0.47]), xaxis2=dict(domain=[0.53, 1]),
                         yaxis2=dict(anchor='x2'), width=width, height=height, 
                         margin=margin, legend=legend)
    fig['layout']['xaxis1'].update(title=feature.capitalize())
    fig['layout']['yaxis1'].update(title='Probability Density')
    fig['layout']['xaxis2'].update(title=feature.capitalize())
    fig['layout']['yaxis2'].update(title='Default Rate')

    return fig


