import numpy as np
import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.data.processing_utils import load_modality, preprocess, df_to_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for processing of raw data")
    parser.add_argument("--data_path", "-P", type=str, required=True, help="Name of experiment")
    parser.add_argument("--test_size", "-TS", type=float, required=True, help="Train/test split ratio")
    parser.add_argument("--split_seed", "-S", type=int, required=True, help="Set a random seed for train/test split")
    parser.add_argument("--truncate_seed", "-T", type=int, required=True, help="Set a random seed for random truncation of undivided examples")
    parser.add_argument("--save_path", "-SP", type=str, required=True, help="Path to save processed data")

    args = parser.parse_args()

    # High-dose dual reporter

    divided_ERK_2, undivided_ERK_2 = load_modality(os.path.join(args.data_path, "high_dose/fig2"), "ERKRatio")
    divided_Akt_2, undivided_Akt_2 = load_modality(os.path.join(args.data_path, "high_dose/fig2"), "AktRatio")
    divided_ERK_s3, undivided_ERK_s3 = load_modality(os.path.join(args.data_path, "high_dose/figs3"), "ERKRatio")
    divided_Akt_s3, undivided_Akt_s3 = load_modality(os.path.join(args.data_path, "high_dose/figs3"), "AktRatio")    

    divided_ERK = np.concatenate([divided_ERK_2, divided_ERK_s3], axis=0)
    undivided_ERK = np.concatenate([undivided_ERK_2, undivided_ERK_s3], axis=0)
    divided_Akt = np.concatenate([divided_Akt_2, divided_Akt_s3], axis=0)
    undivided_Akt = np.concatenate([undivided_Akt_2, undivided_Akt_s3], axis=0)

    divided_ERK_train, divided_ERK_test = train_test_split(divided_ERK, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    undivided_ERK_train, undivided_ERK_test = train_test_split(undivided_ERK, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    divided_Akt_train, divided_Akt_test = train_test_split(divided_Akt, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    undivided_Akt_train, undivided_Akt_test = train_test_split(undivided_Akt, test_size=args.test_size, random_state=args.split_seed, shuffle=True)

    ERK_train, params = preprocess(divided_ERK_train, undivided_ERK_train, length_distribution=None, random_seed=args.truncate_seed)
    Akt_train, _ = preprocess(divided_Akt_train, undivided_Akt_train, length_distribution=None, random_seed=args.truncate_seed)

    ERK_test, _ = preprocess(divided_ERK_test, undivided_ERK_test, length_distribution=params["length distribution"], random_seed=args.truncate_seed)
    Akt_test, _ = preprocess(divided_Akt_test, undivided_Akt_test, length_distribution=params["length distribution"], random_seed=args.truncate_seed)

    df_to_csv(ERK_train, os.path.join(args.save_path, "high_dose/train"), "ERK.csv")
    df_to_csv(Akt_train, os.path.join(args.save_path, "high_dose/train"), "Akt.csv")
    df_to_csv(ERK_test, os.path.join(args.save_path, "high_dose/test"), "ERK.csv")
    df_to_csv(Akt_test, os.path.join(args.save_path, "high_dose/test"), "Akt.csv")

    # Save preprocessing parameters

    path_to_params = os.path.join(args.save_path, "params")
    Path(path_to_params).mkdir(parents=True, exist_ok=True)

    pd.Series(params["length distribution"]).to_csv(os.path.join(path_to_params, "length_distribution.csv"), index=False)

    # Low-dose dual reporter

    divided_ERK_2_low, undivided_ERK_2_low = load_modality(os.path.join(args.data_path, "low_dose/fig2"), "ERKRatio")
    divided_Akt_2_low, undivided_Akt_2_low = load_modality(os.path.join(args.data_path, "low_dose/fig2"), "AktRatio")
    divided_ERK_s3_low, undivided_ERK_s3_low = load_modality(os.path.join(args.data_path, "low_dose/figs3"), "ERKRatio")
    divided_Akt_s3_low, undivided_Akt_s3_low = load_modality(os.path.join(args.data_path, "low_dose/figs3"), "AktRatio")    

    divided_ERK = np.concatenate([divided_ERK_2_low, divided_ERK_s3_low], axis=0)
    undivided_ERK = np.concatenate([undivided_ERK_2_low, undivided_ERK_s3_low], axis=0)
    divided_Akt = np.concatenate([divided_Akt_2_low, divided_Akt_s3_low], axis=0)
    undivided_Akt = np.concatenate([undivided_Akt_2_low, undivided_Akt_s3_low], axis=0)

    ERK_test, _ = preprocess(divided_ERK, undivided_ERK, length_distribution=params["length distribution"], random_seed=args.truncate_seed)
    Akt_test, _ = preprocess(divided_Akt, undivided_Akt, length_distribution=params["length distribution"], random_seed=args.truncate_seed)

    df_to_csv(ERK_test, os.path.join(args.save_path, "low_dose/test"), "ERK.csv")
    df_to_csv(Akt_test, os.path.join(args.save_path, "low_dose/test"), "Akt.csv")

    