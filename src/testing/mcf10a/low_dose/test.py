import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from src.load_data import load_data, make_dicts
from src.dwt import dwt
from src.testing.test_utils import apply_threshold
from eipy.ei import EnsembleIntegration

path_to_data = "data/processed/mcf10a/low_dose/test"
save_path = "results/testing/mcf10a/low_dose"

X_ERK, X_Akt, y, _ = load_data(path_to_data)

X_ERK_t = dwt(X_ERK, wavelet="haar", mode='constant', level=3, axis=-1)[0]
X_Akt_t = dwt(X_Akt, wavelet="haar", mode='constant', level=3, axis=-1)[0]

ERK_Akt, ERK_alone, Akt_alone, y = make_dicts([X_ERK_t, X_Akt_t, y])

modalityIDs = ["ERK/Akt", "ERK", "Akt"]
filenames = ["EI.ERK_Akt", "EI.ERK", "EI.Akt"]
data = [ERK_Akt, ERK_alone, Akt_alone]
modelIDs = ["LR", "RF", "RF"]
save_filenames = ["ERK_Akt.csv", "ERK.csv", "Akt.csv"]

model_dict = dict(zip(modalityIDs, filenames))
data_dict = dict(zip(modalityIDs, data))
modelID_dict = dict(zip(modalityIDs, modelIDs))
save_filenames_dict = dict(zip(modalityIDs, save_filenames))

for modality_id in modalityIDs:
    filename = model_dict[modality_id]
    path = os.path.join("results/models", filename)
    EI = EnsembleIntegration().load(path)
    
    if EI.ensemble_predictors is not None:
        predictions = EI.predict(data_dict[modality_id], modelID_dict[modality_id])
        threshold_f = EI.ensemble_summary["thresholds"][modelID_dict[modality_id]]["fmax (minority)"]

    else:
        all_base_models = deepcopy(EI.final_models["base models"][modality_id])
        base_models = [dictionary for dictionary in all_base_models if dictionary["model name"] == modelID_dict[modality_id]]
        unpickled_base_models = [pickle.loads(base_model["pickled model"]) for base_model in base_models]
        predictions = [model.predict(data_dict[modality_id][modality_id]) for model in unpickled_base_models]
        predictions = np.mean(np.array(predictions), axis=0)
        threshold_f = EI.base_summary["thresholds"].T["fmax (minority)"].to_numpy()

    inference = apply_threshold(predictions, threshold_f)
    df = pd.DataFrame({"inference": inference, "predictions": predictions, "labels": y})
    df.to_csv(os.path.join(save_path, f"predictions_{save_filenames_dict[modality_id]}"), index=False)