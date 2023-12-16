import pandas as pd
import os
from src.ei_setup import initiate_EI
from src.load_data import load_data, make_dicts
from src.dwt import dwt
from sklearn.ensemble import RandomForestClassifier

save_path = f"results/models"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)

data[1] = dwt(data[1], wavelet="haar", mode='constant', level=3, axis=-1)[0]

_, _, Akt, y_train = make_dicts(data)

EI = initiate_EI(model_building=True)

EI.ensemble_predictors = {
                'RF': RandomForestClassifier(max_depth=1),
                }
   
for name, modality in Akt.items():
    EI.fit_base(modality, y_train, modality_name=name)

id = '_'.join(Akt.keys())

EI.fit_ensemble()

EI.save(os.path.join(save_path, "EI." + id))