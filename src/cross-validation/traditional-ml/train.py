import pandas as pd
import os
from src.ei_setup import initiate_EI
from src.load_data import X_ERK, X_Akt, y_train

save_path = f"results/cross-validation/traditional-ml"

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

modality_list = [ERK_Akt, ERK_alone, Akt_alone]

for modalities in modality_list:

    EI = initiate_EI(model_building=False)
   
    for name, modality in modalities.items():
        EI.train_base(modality, y_train, modality=name)

    id = '_'.join(modalities.keys())

    EI.train_meta()

    EI.save(os.path.join(save_path, "EI." + id))