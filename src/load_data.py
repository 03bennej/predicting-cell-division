import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import os

def load_data(path_to_data):
    ERK = pd.read_csv(os.path.join(path_to_data, f"ERK.csv"))
    Akt = pd.read_csv(os.path.join(path_to_data, f"Akt.csv"))

    X_ERK = ERK.drop("label", axis=1)
    y_ERK = ERK["label"] 

    X_Akt = Akt.drop("label", axis=1)
    y_Akt = Akt["label"] 

    data = [X_ERK, X_Akt, y_ERK, y_Akt]

    return data

def load_data_by_filename(path_to_data, filename):
    data = pd.read_csv(os.path.join(path_to_data, filename))

    X = data.drop("label", axis=1)
    y = data["label"] 

    return X, y

def make_dicts(data):

    ERK_Akt = {
                    "ERK": data[0].to_numpy(),
                    "Akt": data[1].to_numpy()
                    }

    ERK_alone = {
                    "ERK": data[0].to_numpy(),
                    }

    Akt_alone = {
                    "Akt": data[1].to_numpy(),
                    }

    y_train = data[2].to_numpy()

    return ERK_Akt, ERK_alone, Akt_alone, y_train