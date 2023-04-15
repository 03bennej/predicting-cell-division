from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from random import choices, seed
from copy import copy
import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path

def standardise_lengths(vector, desired_length):
        vector = vector[:desired_length] 
        vector = np.concatenate([vector, np.zeros(desired_length - vector.shape[0]) * np.nan])    
        return vector

def load_modality(data_path, dosage,  figure, modality):
    df_div = pd.read_csv(os.path.join(data_path, dosage, figure, "divided", modality + ".csv"), header=None)
    df_undiv = pd.read_csv(os.path.join(data_path, dosage, figure, "undivided", modality + ".csv"), header=None)
    return df_div, df_undiv

def random_truncate(divided, undivided, training_lengths=None, random_seed=None):

    divided = copy(divided)
    undivided = copy(undivided)

    if training_lengths is None:
        n_examples_divided = divided.shape[0]
        lengths_divided = []

        for example in range(n_examples_divided):
            lengths_divided.append(len(divided[example, :][~np.isnan(divided[example, :])]))
    else:
        lengths_divided = training_lengths

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

    dataframe = pd.concat([dataframe, labels], axis=1).reset_index(drop=True)

    return dataframe

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for preprocessing of raw data")
    parser.add_argument("--data_path", "-P", type=str, required=True, help="Name of experiment")
    parser.add_argument("--test_size", "-TS", type=float, required=True, help="Train/test split ratio")
    parser.add_argument("--split_seed", "-S", type=int, required=True, help="Set a random seed for train/test split")
    parser.add_argument("--truncate_seed", "-T", type=int, required=True, help="Set a random seed for random truncation of undivided examples")
    parser.add_argument("--save_path", "-SP", type=str, required=True, help="Path to save preprocessed data")

    args = parser.parse_args()

    divided_ERK_2, undivided_ERK_2 = load_modality(args.data_path, "high-dose", "fig2", "ERKRatio")
    divided_Akt_2, undivided_Akt_2 = load_modality(args.data_path, "high-dose", "fig2", "AktRatio")
    divided_ERK_s3, undivided_ERK_s3 = load_modality(args.data_path, "high-dose", "figs3", "ERKRatio")
    divided_Akt_s3, undivided_Akt_s3 = load_modality(args.data_path, "high-dose", "figs3", "AktRatio")    

    divided_ERK = np.concatenate([divided_ERK_2, divided_ERK_s3], axis=0)
    undivided_ERK = np.concatenate([undivided_ERK_2, undivided_ERK_s3], axis=0)
    divided_Akt = np.concatenate([divided_Akt_2, divided_Akt_s3], axis=0)
    undivided_Akt = np.concatenate([undivided_Akt_2, undivided_Akt_s3], axis=0)

    divided_ERK_train, divided_ERK_test = train_test_split(divided_ERK, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    undivided_ERK_train, undivided_ERK_test = train_test_split(undivided_ERK, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    divided_Akt_train, divided_Akt_test = train_test_split(divided_Akt, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    undivided_Akt_train, undivided_Akt_test = train_test_split(undivided_Akt, test_size=args.test_size, random_state=args.split_seed, shuffle=True)

    undivided_ERK_train_trunc, train_lengths_ERK, train_sampled_lengths_ERK = random_truncate(divided_ERK_train, undivided_ERK_train, training_lengths=None, random_seed=args.truncate_seed)
    undivided_Akt_train_trunc, train_lengths_Akt, train_sampled_lengths_Akt = random_truncate(divided_Akt_train, undivided_Akt_train, training_lengths=None, random_seed=args.truncate_seed)
    assert train_lengths_ERK == train_lengths_Akt
    assert train_sampled_lengths_ERK == train_sampled_lengths_Akt

    undivided_ERK_test_trunc, test_lengths_ERK, test_sampled_lengths_ERK = random_truncate(divided_ERK_test, undivided_ERK_test, training_lengths=train_lengths_ERK, random_seed=args.truncate_seed)
    undivided_Akt_test_trunc, test_lengths_Akt, test_sampled_lengths_Akt = random_truncate(divided_Akt_test, undivided_Akt_test, training_lengths=train_lengths_ERK, random_seed=args.truncate_seed)
    assert test_lengths_ERK == test_lengths_Akt
    assert test_sampled_lengths_ERK == test_sampled_lengths_Akt  

    ERK_train = combine_divided_undivided(divided_ERK_train, undivided_ERK_train_trunc)
    ERK_test = combine_divided_undivided(divided_ERK_test, undivided_ERK_test_trunc)
    Akt_train = combine_divided_undivided(divided_Akt_train, undivided_Akt_train_trunc)
    Akt_test = combine_divided_undivided(divided_Akt_test, undivided_Akt_test_trunc)

    ERK_train_pad = mean_pad(ERK_train)
    ERK_test_pad = mean_pad(ERK_test)
    Akt_train_pad = mean_pad(Akt_train)
    Akt_test_pad = mean_pad(Akt_test)

    path_train = os.path.join(args.save_path, "high-dose/train")
    path_test_1 = os.path.join(args.save_path, "high-dose/test")
    Path(path_train).mkdir(parents=True, exist_ok=True)
    Path(path_test_1).mkdir(parents=True, exist_ok=True)

    ERK_train_pad.to_json(os.path.join(path_train, "ERK.json"))
    Akt_train_pad.to_json(os.path.join(path_train, "Akt.json"))
    ERK_test_pad.to_json(os.path.join(path_test_1, "ERK.json"))
    Akt_test_pad.to_json(os.path.join(path_test_1, "Akt.json"))

    divided_ERK_2_low, undivided_ERK_2_low = load_modality(args.data_path, "low-dose", "fig2", "ERKRatio")
    divided_Akt_2_low, undivided_Akt_2_low = load_modality(args.data_path, "low-dose", "fig2", "AktRatio")
    divided_ERK_s3_low, undivided_ERK_s3_low = load_modality(args.data_path, "low-dose", "figs3", "ERKRatio")
    divided_Akt_s3_low, undivided_Akt_s3_low = load_modality(args.data_path, "low-dose", "figs3", "AktRatio")    

    divided_ERK_low = np.concatenate([divided_ERK_2_low, divided_ERK_s3_low], axis=0)
    undivided_ERK_low = np.concatenate([undivided_ERK_2_low, undivided_ERK_s3_low], axis=0)
    divided_Akt_low = np.concatenate([divided_Akt_2_low, divided_Akt_s3_low], axis=0)
    undivided_Akt_low = np.concatenate([undivided_Akt_2_low, undivided_Akt_s3_low], axis=0)

    undivided_ERK_test_trunc_low, test_lengths_ERK_low, test_sampled_lengths_ERK_low = random_truncate(divided_ERK_low, undivided_ERK_low, training_lengths=train_lengths_ERK, random_seed=args.truncate_seed)
    undivided_Akt_test_trunc_low, test_lengths_Akt_low, test_sampled_lengths_Akt_low = random_truncate(divided_Akt_low, undivided_Akt_low, training_lengths=train_lengths_ERK, random_seed=args.truncate_seed)

    ERK_low = combine_divided_undivided(divided_ERK_low, undivided_ERK_test_trunc_low)
    Akt_low = combine_divided_undivided(divided_Akt_low, undivided_Akt_test_trunc_low)

    ERK_low_pad = mean_pad(ERK_low)
    Akt_low_pad = mean_pad(Akt_low)

    path_test_2 = os.path.join(args.save_path, "low-dose/test")
    Path(path_test_2).mkdir(parents=True, exist_ok=True)

    ERK_low_pad.to_json(os.path.join(path_test_2, "ERK.json"))
    Akt_low_pad.to_json(os.path.join(path_test_2, "Akt.json"))