import pandas as pd
import os
from src.ei_setup import initiate_EI
from src.load_data import load_data, make_dicts
from src.dwt import dwt
from sklearn.linear_model import LogisticRegression

save_path = f"results/models"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)

data[0] = dwt(data[0].iloc[:, :], wavelet="haar", mode='constant', level=3, axis=-1)[0]
data[1] = dwt(data[1].iloc[:, :], wavelet="haar", mode='constant', level=3, axis=-1)[0]

ERK_Akt, _, _, y_train = make_dicts(data)

EI = initiate_EI(model_building=True)

EI.ensemble_predictors = {
                'LR': LogisticRegression(),
                }
   
for name, modality in ERK_Akt.items():
    EI.fit_base(modality, y_train, modality_name=name)

id = '_'.join(ERK_Akt.keys())

EI.fit_ensemble()

EI.save(os.path.join(save_path, "EI." + id))