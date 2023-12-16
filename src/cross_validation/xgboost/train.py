import pandas as pd
import numpy as np
import os
from src.xgboost_setup import initiate_EI
from src.load_data import load_data, make_dicts

save_path = "results/cross_validation/xgboost"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)
_, ERK_alone, Akt_alone, y_train = make_dicts(data)

ERK_Akt = {'ERK_Akt': np.concatenate([ERK_alone['ERK'], Akt_alone['Akt']], axis=1)}

modality_list = [ERK_Akt, ERK_alone, Akt_alone]

for modalities in modality_list:

    EI = initiate_EI(model_building=False)
   
    for name, modality in modalities.items():
        EI.fit_base(modality, y_train, modality_name=name)

    id = '_'.join(modalities.keys())

    EI.save(os.path.join(save_path, "EI." + id))