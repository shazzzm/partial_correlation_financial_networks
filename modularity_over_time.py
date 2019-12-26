import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import networkx as nx
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import operator
import matplotlib
import louvain_cython as lcn
from sklearn.metrics import adjusted_rand_score

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

def compare_cluster_consistency(current_assignments, previous_assignments):
    rand_index = []
    for cur_assignment in curr_assignments:
        for prev_assignment in previous_assignments:
            rand_index.append(adjusted_rand_score(cur_assignment, prev_assignment))

    rand_index = np.array(rand_index)
    return np.mean(rand_index), np.std(rand_index), rand_index

np.seterr(all='raise')

#df = pd.DataFrame.from_csv("s_and_p_500_sector_tagged.csv")
df = pd.read_csv("s_and_p_500_daily_close_filtered.csv", index_col=0)
company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
num_sectors = len(sectors)
company_sector_lookup = {}

for i,comp in enumerate(company_names):
    company_sector_lookup[comp] = company_sectors[i]

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

num_runs_community_detection = 10

window_size = 300
slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs-1):
    dates.append(df.index[(x+2)*slide_size+window_size][0:10])

networks_folder = "networks_lw_corr/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder, f)) for f in os.listdir(networks_folder) if os.path.isfile(os.path.join(networks_folder, f))]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
Graphs = []

# Sort the files into order
ind = [int(Path(x).stem[23:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    G = nx.read_graphml(f)
    Graphs.append(G)


max_eigs = np.zeros(no_runs)

sectors = set()
for node in G.nodes:
    sectors.add(G.nodes[node]['sector'])
no_communities = len(sectors)
sectors = list(sectors)
sectors = {sec:i for i,sec in enumerate(sectors)}

sector_assignments_true = np.zeros(len(G.nodes))
rand_scores_all = np.zeros((no_runs, num_runs_community_detection))
rand_scores_mean = []
rand_scores_stdev = []

cluster_consistency_all = np.zeros((no_runs, num_runs_community_detection**2))
cluster_consistency_mean = []
cluster_consistency_stdev = []

for i,node in enumerate(G.nodes):
    sector_assignments_true[i] = sectors[G.nodes[node]['sector']]

prev_assigments = []

number_clusters_all = np.zeros((no_runs, num_runs_community_detection))
number_of_clusters_mean = []
number_of_clusters_stdev = []

nodes = list(G.nodes())

node_assignments = np.chararray(p, itemsize=5)

for i in range(p):
    node_assignments[i] = nodes[i]

np.save("node_assignments", node_assignments)
assignments_overall = np.zeros((no_runs, len(G.nodes), num_runs_community_detection))

for i,G in enumerate(Graphs):
    print("Running %s" % i)
    rand_scores = np.zeros(num_runs_community_detection)
    curr_assignments = []
    num_clusters = np.zeros(num_runs_community_detection)
    for run in range(num_runs_community_detection):
        communities, assignments_dct = lcn.run_louvain_nx(G, nodes, correlation=True)
        assignments = np.zeros(len(G.nodes))
        nodes = list(G.nodes)
        for j,com in enumerate(communities):
            for node in communities[com]:
                assignments[nodes.index(node)] = j
        score = adjusted_rand_score(sector_assignments_true, assignments)
        rand_scores[run] = score
        curr_assignments.append(assignments)
        num_clusters[run] = len(set(assignments))
        assignments_overall[i, :, run] = assignments

    number_clusters_all[i, :] = num_clusters
    number_of_clusters_mean.append(np.mean(num_clusters))
    number_of_clusters_stdev.append(np.std(num_clusters))

    rand_scores_all[i, :]  = rand_scores
    rand_scores_mean.append(np.mean(rand_scores))
    rand_scores_stdev.append(np.std(rand_scores))

    if i > 0:
        consistency_mean, consistency_std, consistency = compare_cluster_consistency(curr_assignments, prev_assigments)
        cluster_consistency_mean.append(consistency_mean)
        cluster_consistency_stdev.append(consistency_std)

        cluster_consistency_all[i, :] = consistency

    prev_assigments = curr_assignments
np.save("overall_assignments", assignments_overall)

np.save(networks_folder[:-1] + "_number_clusters.npy", number_clusters_all)
np.save(networks_folder[:-1] + "_cluster_consistency_all.npy", cluster_consistency_all)
np.save(networks_folder[:-1] + "_rand_scores_all.npy", rand_scores_all)

dt = pd.to_datetime(dates)
ts = pd.Series(rand_scores_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=rand_scores_stdev)
plt.title("Rand Score")
ax.set_ylim(0, 1)

np.save("rand_scores_mean_" + networks_folder[:-1], rand_scores_mean)
np.save("rand_scores_stdev_" +  networks_folder[:-1], rand_scores_stdev)

dt_2 = dt[1:]
ts = pd.Series(cluster_consistency_mean, index=dt_2)
fig = plt.figure()
ax = ts.plot(yerr=cluster_consistency_stdev)
plt.title("Clustering Consistency")
ax.set_ylim(0, 1)

np.save("cluster_consistency_mean_" + networks_folder[:-1], cluster_consistency_mean)
np.save("cluster_consistency_stdev_" + networks_folder[:-1], cluster_consistency_stdev)

ts = pd.Series(number_of_clusters_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=number_of_clusters_stdev)
plt.title("Number of Clusters")
ax.set_ylim(0, 25)

np.save("num_clusters_mean_" + networks_folder[:-1], number_of_clusters_mean)
np.save("num_clusters_stdev_"  + networks_folder[:-1], number_of_clusters_stdev)

save_open_figures("financial_networks_louvain_")
plt.close('all')