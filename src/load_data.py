import pandas as pd
import os

path_to_data = "data/preprocessed/high-dose/train"

ERK = pd.read_json(os.path.join(path_to_data, f"ERK.json"))
Akt = pd.read_json(os.path.join(path_to_data, f"Akt.json"))

X_ERK = ERK.drop("label", axis=1)
y_ERK = ERK["label"] 

X_Akt = Akt.drop("label", axis=1)
y_Akt = Akt["label"] 

ERK_Akt = {
                "ERK": X_ERK.to_numpy(),
                "Akt": X_Akt.to_numpy()
                }

ERK_alone = {
                "ERK": X_ERK.to_numpy(),
                }

Akt_alone = {
                "Akt": X_Akt.to_numpy(),
                }

y_train = y_ERK.to_numpy()