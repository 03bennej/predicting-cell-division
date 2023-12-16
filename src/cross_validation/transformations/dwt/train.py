import pandas as pd
import os
from src.ei_setup import initiate_EI
from src.load_data import load_data, make_dicts
from src.dwt import dwt

save_path = f"results/cross_validation/transformations/dwt"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)

data[0] = dwt(data[0], wavelet="haar", mode='constant', level=3, axis=-1)[0]
data[1] = dwt(data[1], wavelet="haar", mode='constant', level=3, axis=-1)[0]

ERK_Akt, ERK_alone, Akt_alone, y_train = make_dicts(data)

modality_list = [ERK_Akt, ERK_alone, Akt_alone]

for modalities in modality_list:

    EI = initiate_EI(model_building=False)
   
    for name, modality in modalities.items():
        EI.fit_base(modality, y_train, modality_name=name)

    id = '_'.join(modalities.keys())

    EI.fit_ensemble()

    EI.save(os.path.join(save_path, "EI." + id))