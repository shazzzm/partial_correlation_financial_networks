import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
from scipy.stats import norm, spearmanr
import os
import pandas as pd
from pathlib import Path
import operator
import matplotlib
import statsmodels.tsa.stattools
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import multitest
from sklearn.covariance import LedoitWolf

def get_centrality(G, degree=True):
    """
    Calculates the centrality of each node and mean centrality of a sector 
    if degree is true we use degree centrality, if not we use eigenvector centrality
    """
    node_centrality = collections.defaultdict(float)
    total = 0

    if not degree:
        # Do eigenvector centrality
        M = nx.to_numpy_matrix(G)
        _, eigv = scipy.linalg.eigh(prec, eigvals=(p-1, p-1))
        total = eigv.sum()
        for i,node in enumerate(G.nodes):
            node_centrality[node] = eigv[i][0]/total
    else:
        # Calculate the weighted edge centrality
        for node in G.nodes:
            for edge in G[node]:
                node_centrality[node] += G[node][edge]['weight']
                total += G[node][edge]['weight']

        # Normalise so the total is 1
        for comp in node_centrality:
            node_centrality[comp] = node_centrality[comp]/total

    sorted_centrality = sort_dict(node_centrality)
    centrality_names = [x[0] for x in sorted_centrality]
    centrality_sectors = []

    for name in centrality_names:
        centrality_sectors.append(G.nodes[name]['sector'])

    # Figure out the mean centrality of a sector
    sector_centrality = collections.defaultdict(float)
    no_companies_in_sector = collections.defaultdict(int)

    for comp in G:
        sector = G.nodes[comp]['sector']
        sector_centrality[sector] += node_centrality[comp]
        no_companies_in_sector[sector] += 1
    for sec in sector_centrality:
        sector_centrality[sec] /= no_companies_in_sector[sec]

    return node_centrality, sector_centrality

def turn_dict_into_np_array(dct, company_names):
    """
    Turns the dct into a numpy array where the keys are held in company_names
    """
    company_names = list(company_names)
    ret_arr = np.zeros(len(company_names))
    for key in dct:
        i = company_names.index(key)
        ret_arr[i] = dct[key]

    return ret_arr

def sort_dict(dct):
    """
    Takes a dict and returns a sorted list of key value pairs
    """
    sorted_x = sorted(dct.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_x

def save_open_figures(prefix=""):
    """
    Saves all open figures
    """
    figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

    for i, figure in enumerate(figures):
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        figure.savefig(prefix+'figure%d.png' % i)

def threshold_matrix(M, threshold):
    """
    Turns values below threshold to 0 and above threshold to 1
    """
    A = M.copy()
    low_value_indices = np.abs(A) < threshold
    A[low_value_indices] = 0
    high_value_indices = np.abs(A) > threshold
    A[high_value_indices] = 1
    return A


def get_sector_full_nice_name(sector):
    """
    Returns a short version of the sector name
    """       
    if sector == "information_technology":
        return "Information Technology"
    elif sector == "real_estate":
        return "Real Estate"
    elif sector == "materials":
        return "Materials"
    elif sector == "telecommunication_services":
        return "Telecommunication Services"
    elif sector == "energy":
        return "Energy"
    elif sector == "financials":
        return "Financials"
    elif sector == "utilities":
        return "Utilities"
    elif sector == "industrials":
        return "Industrials"
    elif sector == "consumer_discretionary":
        return "Consumer Discretionary"
    elif sector == "health_care":
        return "Healthcare"
    elif sector == "consumer_staples":
        return "Consumer Staples"
    else:
        raise Exception("%s is not a valid sector" % sector)

def plot_bar_chart(vals, label=None, title=None, xlabel=None, ylabel=None):
    fig = plt.figure()
    n = vals.shape[0]
    index = np.arange(n)
    bar_width = 0.1
    rects1 = plt.bar(index, vals, bar_width, label=label)
    #axes = fig.axes
    #print(axes)
    #axes[0].set_xticklabels(label)
    plt.xticks(index, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

np.seterr(all='raise')
df = pd.read_csv("s_and_p_500_daily_close_filtered.csv", index_col=0)
company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
num_sectors = len(sectors)
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

window_size = 300
slide_size = 30
no_samples = X.shape[0]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df.index[(x+1)*slide_size+window_size][0:10])

dt = pd.to_datetime(dates)
"""
networks_folder_correlation = "networks_lw_corr/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder_correlation, f)) for f in os.listdir(networks_folder_correlation) if os.path.isfile(os.path.join(networks_folder_correlation, f))]
#onlyfiles = onlyfiles[0:1]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
Graphs_correlation = []

# Sort the files into order
ind = [int(Path(x).stem[23:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    G = nx.read_graphml(f)
    Graphs_correlation.append(G)


"""
networks_folder_partial_correlation = "networks_lw/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder_partial_correlation, f)) for f in os.listdir(networks_folder_partial_correlation) if os.path.isfile(os.path.join(networks_folder_partial_correlation, f))]
#onlyfiles = onlyfiles[0:1]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
Graphs_partial_correlation = []

# Sort the files into order
#ind = [int(Path(x).stem[18:]) for x in onlyfiles]
ind = [int(Path(x).stem[23:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    G = nx.read_graphml(f)
    Graphs_partial_correlation.append(G)

number_graphs = len(Graphs_partial_correlation)
number_companies = len(Graphs_partial_correlation[0])
p = number_companies

degree_centrality_par_corr = np.zeros((no_runs, p))
optimal_portfolio_diff_par_corr = np.zeros(no_runs)
precision_diag_sum = np.zeros(no_runs)

for i in range(number_graphs):
    X_new = X[i*slide_size:(i+1)*slide_size+window_size, :]
    lw = LedoitWolf()
    lw.fit(X_new)
    prec = lw.precision_
    precision_diag_sum[i] = np.diag(prec).sum()
    optimal_portfolio = (1/ (np.ones(p).T @ prec @ np.ones(p))) * prec @ np.ones(p) 
    prec = np.array(nx.to_numpy_matrix(Graphs_partial_correlation[i]))

    degree_centrality_prec = prec.copy()
    degree_centrality = prec.sum(axis=0)
    degree_centrality /= degree_centrality.sum()

    optimal_portfolio_diff_par_corr[i] = np.linalg.norm(degree_centrality - optimal_portfolio)

optimal_portfolio_degree_centrality_diff = pd.DataFrame()
#ts = pd.Series(optimal_portfolio_diff_corr, index=dt)
#optimal_portfolio_degree_centrality_diff['Correlation'] = ts
ts = pd.Series(optimal_portfolio_diff_par_corr, index=dt)
optimal_portfolio_degree_centrality_diff['Partial Correlation'] = ts
optimal_portfolio_degree_centrality_diff.plot(legend=False)
plt.title("Optimal Portfolio vs Degree Centrality Diff")
plt.savefig("optimal_portfolio_vs_degree_centrality.png")

plt.figure()
ts = pd.Series(precision_diag_sum, index=dt)
ts.plot()
plt.title("Precision Matrix Diagonal Sum")
plt.savefig("precision_matrix_diagonal_sum.png")

plt.show()
#save_open_figures("financial_networks_portfolio_")
#plt.close('all')
