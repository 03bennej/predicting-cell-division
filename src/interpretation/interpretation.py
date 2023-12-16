import numpy as np
import os
import pandas as pd
import pickle
from src.load_data import load_data, make_dicts
from src.dwt import dwt
from eipy.ei import EnsembleIntegration
from eipy.interpretation import PermutationInterpreter
from eipy.metrics import fmax_score

def LR_ranks(LR_model):
    LR_coeffs = pd.DataFrame(np.abs(LR_stacker.coef_[0]))
    return LR_coeffs.rank(pct=True, ascending=False)

save_path = f"results/interpretation"
path_to_models = f"results/models"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)
data[0] = dwt(data[0], wavelet="haar", mode='constant', level=3, axis=-1)[0]
data[1] = dwt(data[1], wavelet="haar", mode='constant', level=3, axis=-1)[0]
ERK_Akt, _, _, y_train = make_dicts(data)

EI = EnsembleIntegration().load(os.path.join(path_to_models, "EI.ERK_Akt"))

LR_stacker = pickle.loads(EI.final_models["ensemble models"]["LR"])

LR_coeffs = LR_ranks(LR_stacker)

id = '_'.join(ERK_Akt.keys())

EI_int = PermutationInterpreter(EI=EI,
                                metric=lambda y_test, y_pred: fmax_score(y_test, y_pred)[0], 
                                n_repeats=100,
                                ensemble_predictor_keys=["LR"],
                                n_jobs=-5)

EI_int.rank_product_score(ERK_Akt, y_train)

EI_int.LMR["LMR"] = LR_coeffs

EI_int.rank_product_score(ERK_Akt, y_train)  # calculate rps with LR coeffs instead

EI_int.LFR.to_csv(os.path.join(save_path, "LFR_" + id + ".csv" ))

LR_coeffs.to_csv(os.path.join(save_path, "LR_coeffs_" + id + ".csv" ))

EI_int.ensemble_feature_ranking['LR'].to_csv(os.path.join(save_path, "interpretation_" + id + ".csv" ))