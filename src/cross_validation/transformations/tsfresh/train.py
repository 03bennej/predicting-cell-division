import pandas as pd
import os
from src.ei_setup import initiate_EI
from src.load_data import load_data, make_dicts
import tsfresh

save_path = f"results/cross_validation/transformations/tsfresh"
data_path = "data/processed/mcf10a/high_dose/train"

data = load_data(data_path)

data[0] = tsfresh.extract_features(pd.melt(data[0].T), column_id="variable", column_kind=None, column_value=None)
data[1] = tsfresh.extract_features(pd.melt(data[1].T), column_id="variable", column_kind=None, column_value=None)

data[0] = data[0].dropna(axis='columns')
data[1] = data[1].dropna(axis='columns')

nunique = data[0].nunique()
cols_to_drop = nunique[nunique == 1].index
data[0] = data[0].drop(cols_to_drop, axis=1)

nunique = data[1].nunique()
cols_to_drop = nunique[nunique == 1].index
data[1] = data[1].drop(cols_to_drop, axis=1)

ERK_Akt, ERK_alone, Akt_alone, y_train = make_dicts(data)

modality_list = [ERK_Akt, ERK_alone, Akt_alone]

for modalities in modality_list:

    EI = initiate_EI(model_building=False)
   
    for name, modality in modalities.items():
        EI.fit_base(modality, y_train, modality_name=name)

    id = '_'.join(modalities.keys())

    EI.fit_ensemble()

    EI.save(os.path.join(save_path, "EI." + id))