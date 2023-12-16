import pandas as pd
import os
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.transformations.panel.rocket import MiniRocket
from src.ei_setup import initiate_EI
from src.load_data import load_data, make_dicts

save_path = f"results/cross_validation/transformations/minirocket"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)

X_ERK_nested = from_2d_array_to_nested(data[0])
X_Akt_nested = from_2d_array_to_nested(data[1])

transform_ERK = MiniRocket(random_state=42)
transform_Akt = MiniRocket(random_state=42)
transform_ERK.fit(X_ERK_nested)
transform_Akt.fit(X_Akt_nested)

data[0] = transform_ERK.transform(X_ERK_nested)
data[1] = transform_Akt.transform(X_Akt_nested)

ERK_Akt, ERK_alone, Akt_alone, y_train = make_dicts(data)

modality_list = [ERK_Akt, ERK_alone, Akt_alone]

for modalities in modality_list:

    EI = initiate_EI(model_building=False)
   
    for name, modality in modalities.items():
        EI.fit_base(modality, y_train, modality_name=name)

    id = '_'.join(modalities.keys())

    EI.fit_ensemble()

    EI.save(os.path.join(save_path, "EI." + id))