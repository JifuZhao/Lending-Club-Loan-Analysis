#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility functions """


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools

warnings.simplefilter('ignore')


def plot_numerical(data, feature, figsize=(16, 5)):
    """ helper function for visualization using Seaborn  """
    data = data[~data[feature].isnull()]
    grouped = data[[feature, 'target']].groupby(feature)
    mean = grouped.mean().reset_index()
    hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}
    warnings.filterwarnings('ignore')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.distplot(data[data['target'] == 0][feature], label='Target: 0',
                 ax=ax[0], hist_kws=hist_kws)
    sns.distplot(data[data['target'] == 1][feature], label='Target: 1',
                 ax=ax[0], hist_kws=hist_kws)
    ax[0].legend()
    ax[1].plot(mean[feature], mean['target'], '.:', alpha=0.5)
    ax[1].set_xlabel(feature)
    ax[1].set_ylabel('Mean')
    ax[1].grid(True)
    plt.tight_layout()
    return fig, ax


def discrete_plot(data, feature, width=800, height=400):
    """ function to plot the discrete variable with Plotly """
    # make subplots
    titles = ('Distribution Plot of ' + feature.capitalize(),
              'Default Rate vs. '+ feature.capitalize())
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)

    # fig 1: count distribution for each feature
    grouped = data.groupby('target')[feature]
    values = grouped.apply(lambda x: x.value_counts(normalize=True)).unstack()

    trace0 = go.Bar(x=values.columns, y=values.loc[0], name='Status 0')
    trace1 = go.Bar(x=values.columns, y=values.loc[1], name='Status 1')
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)

    # fig 2: default rate bar plot for each feature
    names = list(values.columns)
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
    fig['layout']['xaxis1'].update(title=feature.capitalize())
    fig['layout']['yaxis1'].update(title='Probability Density')
    fig['layout']['xaxis2'].update(title=feature.capitalize())
    fig['layout']['yaxis2'].update(title='Default Rate')

    return fig


def numerical_plot(data, feature, hist_bins=40, scatter_bins=100, log=False, w=1000, h=450):
    """ function to plot the numerical variable with Plotly """
    # transform into log scale
    if log is True:
        data = data.copy()
        tail = ' (log)'
        if np.min(data[feature]) == 0:
            data[feature] = np.log(data[feature] + 1)
        data[feature] = np.log(data[feature] + 1)
    else:
        tail = ''

    # make subplots
    titles = ('Histogram of ' + feature.capitalize() + tail,
              'Default Rate vs. ' + feature.capitalize() + tail)
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)

    # fig 1: histogram for different loan status
    x0 = data[data['target']==0][feature]
    x1 = data[data['target']==1][feature]

    # find the minimum and maximum values
    start = min(x0.min(), x1.min())
    end = max(x0.max(), x1.max())
    n_unique = len(data[feature].unique())
    if n_unique <= min(end - start + 1, hist_bins):
        bin_size = 1
    else:
        bin_size = (end - start) / hist_bins

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

    # fig 2: scatter plot for each feature
    cut = pd.cut(data[feature], bins=scatter_bins)
    group_median = data[[feature, 'target']].groupby(cut).median()
    index = group_median[~group_median[feature].isnull()][feature].values
    grouped_mean = data[[feature, 'target']].groupby(cut).mean().fillna(method='pad')
    mean = grouped_mean[~group_median[feature].isnull()]['target'].values
    grouped_sem = data[[feature, 'target']].groupby(cut).sem().fillna(method='pad')
    sem = grouped_sem[~group_median[feature].isnull()]['target'].values

    # make figures
    lower = go.Scatter(name='Lower Bound', x=index, y=mean - sem, mode='lines',
                       marker=dict(color="#444"), line=dict(width=0), showlegend=False)

    trace = go.Scatter(name='Default Rate', x=index, y=mean, mode='lines',
                       line=dict(color='rgb(31, 119, 180)', width=1),
                       fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty')

    upper = go.Scatter(name='Upper Bound', x=index, y=mean + sem, mode='lines',
                       marker=dict(color="#444"), line=dict(width=0), fill='tonexty',
                       fillcolor='rgba(68, 68, 68, 0.3)', showlegend=False)

    fig.append_trace(lower, 1, 2)
    fig.append_trace(trace, 1, 2)
    fig.append_trace(upper, 1, 2)

    # layout setting
    legend = dict(orientation='h', xanchor='auto', y=-0.2)
    margin=go.layout.Margin(l=50, r=50, b=50, t=40, pad=4)
    fig['layout'].update(xaxis=dict(domain=[0, 0.47]), xaxis2=dict(domain=[0.53, 1]),
                         yaxis2=dict(anchor='x2'), width=w, height=h,
                         margin=margin, legend=legend)
    fig['layout']['xaxis1'].update(title=feature.capitalize() + tail)
    fig['layout']['yaxis1'].update(title='Probability Density')
    fig['layout']['xaxis2'].update(title=feature.capitalize() + tail)
    fig['layout']['yaxis2'].update(title='Default Rate')

    return fig


# def numerical_plot(data, feature, width=800, height=400, bins=50):
#     """ function to plot the numerical variable """
#     # make subplots
#     titles = ('Histogram Plot', 'Default Rate vs. ' + feature.capitalize())
#     fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)
#
#     # fig 1: histogram for different loan status
#     x0 = data[data['target']==0][feature]
#     x1 = data[data['target']==1][feature]
#
#     # find the minimum and maximum values
#     start = min(x0.min(), x1.min())
#     end = max(x0.max(), x1.max())
#     n_unique = len(data[feature].unique())
#     if n_unique <= min(end - start + 1, bins):
#         bin_size = 1
#     else:
#         bin_size = (end - start) / bins
#
#     # Group data together
#     hist_data = [x0, x1]
#     group_labels = ['Status 0', 'Status 1']
#
#     # Create distplot
#     fig1 = ff.create_distplot(hist_data=hist_data, group_labels=group_labels,
#                               bin_size=bin_size, show_rug=False)
#     displot = fig1['data']
#
#     # add histgram into the final figure
#     fig.append_trace(displot[0], 1, 1)
#     fig.append_trace(displot[1], 1, 1)
#     fig.append_trace(displot[2], 1, 1)
#     fig.append_trace(displot[3], 1, 1)
#
#     # fig 2: scatter plot for each feature
#     mean = train.groupby(feature)['target'].mean()
#     sem = train.groupby(feature)['target'].sem().fillna(value=0)
#     index = mean.index
#
#     lower = go.Scatter(x=index, y=mean[index]-sem[index], mode='lines',
#                        marker=dict(color="#444"), line=dict(width=0),
#                        showlegend=False)
#
#     trace = go.Scatter(name='Default Rate', x=index, y=mean[index],
#                        line=dict(color='rgb(31, 119, 180)', width=1),
#                        fillcolor='rgba(68, 68, 68, 0.3)', mode='lines',)
#
#     upper = go.Scatter(x=index, y=mean[index]+sem[index], mode='lines',
#                        marker=dict(color="#444"), line=dict(width=0),
#                        fill='tonexty', fillcolor='rgba(68, 68, 68, 0.3)',
#                        showlegend=False)
#
#     fig.append_trace(lower, 1, 2)
#     fig.append_trace(upper, 1, 2)
#     fig.append_trace(trace, 1, 2)
#
#     # layout setting
#     legend = dict(orientation='h', xanchor='auto', y=-0.2)
#     margin=go.layout.Margin(l=50, r=50, b=50, t=40, pad=4)
#     fig['layout'].update(xaxis=dict(domain=[0, 0.47]), xaxis2=dict(domain=[0.53, 1]),
#                          yaxis2=dict(anchor='x2'), width=width, height=height,
#                          margin=margin, legend=legend)
#     fig['layout']['xaxis1'].update(title=feature.capitalize())
#     fig['layout']['yaxis1'].update(title='Probability Density')
#     fig['layout']['xaxis2'].update(title=feature.capitalize())
#     fig['layout']['yaxis2'].update(title='Default Rate')
#
#     return fig
#
#
# def categorical_plot(data, feature, width=800, height=400):
#     """ function to plot the categorical variable """
#     # make subplots
#     titles = ('Distribution Plot', 'Default Rate Distribution')
#     fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=titles)
#
#     # fig 1: count distribution for each feature
#     grouped = data.groupby('target')[feature]
#     values = grouped.apply(lambda x: x.value_counts(normalize=True)).unstack()
#     names = list(values.columns)
#     x = ['status 0', 'status 1']
#     for name in names:
#         trace = go.Bar(x=x, y=list(values[name]), name=name)
#         fig.append_trace(trace, 1, 1)
#
#     # fig 2: default rate bar plot for each feature
#     means = data.groupby(feature)['target'].mean()
#     stds = data.groupby(feature)['target'].std()
#     for name, mean, std in zip(names, means[names], stds[names]):
#         low, high = stats.norm.interval(0.05, loc=mean, scale=std)
#         er = mean - low
#         trace = go.Bar(x=[name], y=[mean], error_y=dict(array=[er], visible=True),
#                        name=name, xaxis='x2')
#         fig.append_trace(trace, 1, 2)
#
#     # layout setting
#     legend = dict(orientation='h', xanchor='auto', y=-0.2)
#     margin=go.layout.Margin(l=50, r=50, b=50, t=40, pad=4)
#     fig['layout'].update(xaxis=dict(domain=[0, 0.47]), xaxis2=dict(domain=[0.53, 1]),
#                          yaxis2=dict(anchor='x2'), width=width, height=height,
#                          margin=margin, legend=legend)
#     fig['layout']['xaxis1'].update(title='Loan Status')
#     fig['layout']['yaxis1'].update(title='Probability Density')
#     fig['layout']['xaxis2'].update(title=feature.capitalize())
#     fig['layout']['yaxis2'].update(title='Default Rate')
#
#     return fig
