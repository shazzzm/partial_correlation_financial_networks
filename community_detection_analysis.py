import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from scipy.stats import distributions
import pandas as pd
import math

def ttest_ind(a_mean, a_var, b_mean, b_var, axis=0, equal_var=True):
    v1 = a_var
    v2 = b_var
    n1 = 5
    n2 = 5

    if (equal_var):
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, axis) - np.mean(b, axis)
    t = np.divide(d, denom)
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail

    return t, prob

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

window_size = 300
slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df.index[(x+1)*slide_size+window_size][0:10])
dt = pd.to_datetime(dates)

# Read in the correlation and partial correlation results
cluster_consistency_mean_corr = np.load("cluster_consistency_mean_corr.npy")
cluster_consistency_mean_par_corr = np.load("cluster_consistency_mean_par_corr.npy")
cluster_consistency_stdev_corr = np.load("cluster_consistency_stdev_corr.npy")
cluster_consistency_stdev_par_corr = np.load("cluster_consistency_stdev_par_corr.npy")

num_clusters_mean_corr = np.load("num_clusters_mean_corr.npy")
num_clusters_mean_par_corr = np.load("num_clusters_mean_par_corr.npy")
num_clusters_stdev_corr = np.load("num_clusters_stdev_corr.npy")
num_clusters_stdev_par_corr = np.load("num_clusters_stdev_par_corr.npy")

rand_scores_mean_corr = np.load("rand_scores_mean_corr.npy")
rand_scores_mean_par_corr = np.load("rand_scores_mean_par_corr.npy")
rand_scores_stdev_corr = np.load("rand_scores_stdev_corr.npy")
rand_scores_stdev_par_corr = np.load("rand_scores_stdev_par_corr.npy")

rand_scores_mean = pd.DataFrame()
rand_scores_stdev = pd.DataFrame()
ts = pd.Series(rand_scores_mean_corr, index=dt)
rand_scores_mean['Correlation'] = ts
ts = pd.Series(rand_scores_mean_par_corr, index=dt)
rand_scores_mean['Partial Correlation'] = ts

ts = pd.Series(rand_scores_stdev_corr, index=dt)
rand_scores_stdev['Correlation'] = ts
ts = pd.Series(rand_scores_stdev_par_corr, index=dt)
rand_scores_stdev['Partial Correlation'] = ts

ax = rand_scores_mean.plot(yerr=rand_scores_stdev)
plt.title("Rand Score")
#ax.set_ylim(0, 1)

number_clusters_mean_df = pd.DataFrame()
number_clusters_stdev_df = pd.DataFrame()
ts = pd.Series(num_clusters_mean_corr, index=dt)
number_clusters_mean_df['Correlation'] = ts
ts = pd.Series(num_clusters_mean_par_corr, index=dt)
number_clusters_mean_df['Partial Correlation'] = ts

ts = pd.Series(num_clusters_stdev_corr, index=dt)
number_clusters_stdev_df['Correlation'] = ts
ts = pd.Series(num_clusters_stdev_par_corr, index=dt)
number_clusters_stdev_df['Partial Correlation'] = ts

ax = number_clusters_mean_df.plot(yerr=number_clusters_stdev_df)
plt.title("Mean Number of Clusters")

cluster_consistency_mean_df = pd.DataFrame()
cluster_consistency_stdev_df = pd.DataFrame()
ts = pd.Series(cluster_consistency_mean_corr, index=dt_2)
cluster_consistency_mean_df['Correlation'] = ts
ts = pd.Series(cluster_consistency_mean_par_corr, index=dt_2)
cluster_consistency_mean_df['Partial Correlation'] = ts

ts = pd.Series(cluster_consistency_stdev_corr, index=dt_2)
cluster_consistency_stdev_df['Correlation'] = ts
ts = pd.Series(cluster_consistency_stdev_par_corr, index=dt_2)
cluster_consistency_stdev_df['Partial Correlation'] = ts

ax = cluster_consistency_mean_df.plot(yerr=cluster_consistency_stdev_df)
plt.title("Clustering Consistency")

plt.show()