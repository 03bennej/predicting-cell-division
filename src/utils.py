import pandas as pd
import os
import numpy as np
from eipy.ei import EnsembleIntegration
import matplotlib.pyplot as plt
import seaborn
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ranksums

results_path = "predicting-cell-division/results"

def load_ei_results(path):
    EI = EnsembleIntegration().load(path)
    bs_df = EI.base_summary["metrics"].T
    meta_df = EI.ensemble_summary
    if meta_df is not None:
        meta_df = EI.ensemble_summary["metrics"].T
    return bs_df, meta_df

def best_method(df, scoring_method="fmax (minority)"):
    index = df[scoring_method].idxmax()
    return df.iloc[index, :]

def f_baseline(y):
    ones = len(y[y == 1])
    zeros = len(y[y == 0])  
    class_imbalance = ones / (ones + zeros)  
    return 2*class_imbalance/(class_imbalance + 1)

def tsplot(ax, x, data, color=None, label=None):
    est = np.nanmedian(data, axis=0)
    mad = np.median(np.abs(data - est))
    # sd = np.nanstd(data, axis=0)
    cis = (est - mad, est + mad)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=color)
    ax.plot(x, est, color=color, label=label)
    ax.margins(x=0)

def pairwise_rank_sum(df):
    a_groups = df.columns.to_list()
    b_groups = df.columns.to_list()
    num_groups = len(a_groups)
    pairwise_mat = -1*np.ones((num_groups, num_groups))
    idx_list = []
    pval_list = []
    print(df)
    for idx_a, a in enumerate(a_groups):
        for idx_b, b in enumerate(b_groups):
            if idx_a != idx_b:
                idx_list.append([idx_a, idx_b])
                # print(df[a].values, df[b].values)
                pval_list.append(ranksums(df[a].values, df[b].values, alternative='greater')[1])
                # pairwise_mat[idx_a, idx_b] = ranksums(df[a].values, df[b].values).pvalue
                # pairwise_mat[idx_b, idx_a] = ranksums(df[a].values, df[b].values).pvalue
    # print('p_val:', pval_list)
    pval_list = fdrcorrection(pval_list)[1]
    # print('fdr_corrected:',pval_list)
    for idx, pval in zip(idx_list, pval_list):
        pairwise_mat[idx[0], idx[1]] = pval
        # pairwise_mat[idx[1], idx[0]] = pval
    return pd.DataFrame(pairwise_mat, index=a_groups, columns=b_groups)


