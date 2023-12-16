import pandas as pd
import numpy as np
import os
from copy import copy
from pathlib import Path
from random import choices, seed

def standardise_lengths(vector, desired_length):
        vector = vector[:desired_length] 
        vector = np.concatenate([vector, np.zeros(desired_length - vector.shape[0]) * np.nan])    
        return vector

def load_modality(data_path, modality):
    df_div = pd.read_csv(os.path.join(data_path, "divided", modality + ".csv"), header=None)
    df_undiv = pd.read_csv(os.path.join(data_path, "undivided", modality + ".csv"), header=None)
    return df_div, df_undiv

def random_truncate(divided, undivided, length_distribution=None, random_seed=None):

    divided = copy(divided)
    undivided = copy(undivided)

    if length_distribution is None:
        n_examples_divided = divided.shape[0]
        lengths_divided = []

        for example in range(n_examples_divided):
            lengths_divided.append(len(divided[example, :][~np.isnan(divided[example, :])]))
    else:
        lengths_divided = length_distribution

    seed(random_seed)
    sampled_lengths = choices(lengths_divided, k=undivided.shape[0]) 
        
    undivided_trunc = np.empty((undivided.shape[0], divided.shape[1]))

    for i in range(undivided_trunc.shape[0]):
        truncated_example = undivided[i, :sampled_lengths[i]]
        pad_length = divided.shape[1] - len(truncated_example)
        undivided_trunc[i, :] = np.pad(truncated_example, (0, pad_length), 'constant', constant_values=np.nan)

    assert undivided_trunc.shape[1] == divided.shape[1]

    return undivided_trunc, lengths_divided, sampled_lengths

def combine_divided_undivided(divided, undivided):

    divided = pd.DataFrame(divided)
    undivided = pd.DataFrame(undivided)

    divided["label"] = 1
    undivided["label"] = 0

    df = pd.concat([divided, undivided])

    return df

def mean_pad(dataframe):

    dataframe = copy(dataframe)
    labels = dataframe["label"]
    dataframe = dataframe.drop(["label"], axis=1)
    for example_id in range(dataframe.shape[0]):
        dataframe.iloc[example_id, :][np.isnan(dataframe.iloc[example_id, :])] = np.nanmean(dataframe.iloc[example_id, :]) 
        if np.isnan(dataframe.iloc[example_id, :].to_numpy()).any():
            print("\n", "Example ", example_id, "is all NaNs", "\n")

    dataframe = pd.concat([dataframe, labels], axis=1).reset_index(drop=True)

    return dataframe

def preprocess(X_divided, X_undivided, length_distribution, random_seed=None):

    X_undivided_trunc, length_distribution, sampled_lengths = random_truncate(X_divided, X_undivided, length_distribution=length_distribution, random_seed=random_seed)
    data = combine_divided_undivided(X_divided, X_undivided_trunc)

    data_pad = mean_pad(data)

    params = {"length distribution": length_distribution,
              "sampled lengths": sampled_lengths}

    return data_pad, params

def df_to_csv(df, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(path, name), index=False)