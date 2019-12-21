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
from statsmodels.stats import multitest
import itertools

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

number_graphs = len(Graphs_correlation)
number_companies = len(Graphs_correlation[0])
p = number_companies
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

par_corr_vals = np.zeros((number_companies ** 2 * 140))
corr_vals = np.zeros((number_companies ** 2 * 140))
correlation_eigv_diff = np.zeros(number_graphs-1)
partial_correlation_eigv_diff = np.zeros(number_graphs-1)
corr_par_corr_diff = np.zeros(number_graphs)
corr_eigv = np.zeros((number_graphs, p))
par_corr_eigv = np.zeros((number_graphs, p))


for i in range(number_graphs):
    correlation = nx.to_numpy_array(Graphs_correlation[i])
    par_corr = nx.to_numpy_array(Graphs_partial_correlation[i])
    corr_vals[i*(p**2):(i+1)*(p**2)] = correlation.flatten()
    par_corr_vals[i*(p**2):(i+1)*(p**2)] = par_corr.flatten()

    corr_par_corr_diff[i] = spearmanr(correlation.flatten(), par_corr.flatten())[0]
    eigs, eigv = scipy.linalg.eigh(correlation, eigvals=(p-1, p-1))
    eigv = eigv/eigv.sum()
    corr_eigv[i, :] = eigv.flatten()
    if i > 0:
        correlation_eigv_diff[i-1] = np.linalg.norm(corr_eigv[i-1,:] - eigv)

    eigs, eigv = scipy.linalg.eigh(par_corr, eigvals=(p-1, p-1))
    eigv = eigv/eigv.sum()
    par_corr_eigv[i, :] = eigv.flatten()
    if i > 0:
        partial_correlation_eigv_diff[i-1] = np.linalg.norm(par_corr_eigv[i-1,:] - eigv)

#for i in range(number_graphs-1):
#    corr_par_corr_kendall_tau[i] = scipy.stats.kendalltau(largest_corr_par_corr_diff[:, i+1], largest_corr_par_corr_diff[:, i])[0]

plt.scatter(corr_vals, par_corr_vals)
plt.xlabel("Correlation")
plt.ylabel("Partial Correlation")
plt.savefig("correlation_vs_partial_correlation.png")

eigv_diff = pd.DataFrame()
ts = pd.Series(correlation_eigv_diff, index=dt[1:])
eigv_diff["Correlation"] = correlation_eigv_diff
ts = pd.Series(partial_correlation_eigv_diff, index=dt[1:])
eigv_diff["Partial Correlation"] = partial_correlation_eigv_diff
eigv_diff.index = dt[1:]

#plt.figure()
eigv_diff.plot()
plt.title("Largest Eigenvector Diff")
ax = plt.gca()
ax.set_ylim(0, 1)

plt.show()
