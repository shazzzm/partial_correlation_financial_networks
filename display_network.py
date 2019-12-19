import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
from scipy.stats import norm, spearmanr
import os
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import operator
import matplotlib
import statsmodels.tsa.stattools
from sklearn.preprocessing import StandardScaler
import modularity_maximizer
from statsmodels.stats import multitest

def threshold_graph(G):
    M = nx.to_numpy_matrix(G)
    # Select 5000 edges 
    np.fill_diagonal(M, 0)
    vals = np.array(M.flatten()).flatten()
    vals = np.sort(np.abs(vals))[::-1]
    threshold = vals[1999] 
    M[np.abs(M) < threshold] = 0
    G_new = nx.from_numpy_matrix(M)
    mapping = { x : node for x, node in enumerate(G)}
    G_new = nx.relabel_nodes(G_new, mapping)

    for n in G.nodes:
        G_new.nodes[n]['sector'] = G.nodes[n]['sector']

    nx.write_graphml(G_new, "partial_correlation.graphml")
    return G_new


# Change this if you wish to analyze either correlation or partial correlation networks
networks_folder = "networks_lw/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder, f)) for f in os.listdir(networks_folder) if os.path.isfile(os.path.join(networks_folder, f))]
#onlyfiles = onlyfiles[0:1]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
Graphs = []

# Sort the files into order
ind = [int(Path(x).stem[23:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    G = nx.read_graphml(f)
    Graphs.append(G)

number_graphs = len(Graphs)

G = Graphs[0]
threshold_graph(G)
