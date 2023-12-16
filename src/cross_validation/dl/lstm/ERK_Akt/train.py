from sklearn.preprocessing import StandardScaler
from src.dl_setup import LSTM, compile_kwargs, inner_fit_kwargs
from src.dl_cv import stack_modalities, dl_cv, n_outer, n_inner, random_state
from src.load_data import load_data
import numpy as np
import pandas as pd
import os

save_path = "results/cross_validation/dl/lstm"
data_path = "data/preprocessed/high_dose/train"

data = load_data(data_path)
X_ERK = data[0]
X_Akt = data[1]
y_train = data[2]

scaler = StandardScaler()

X_ERK = scaler.fit_transform(X_ERK)
X_Akt = scaler.fit_transform(X_Akt)

X = stack_modalities(X_ERK, X_Akt)

model = LSTM(shape=(X.shape[1], X.shape[2]))

y_pred, y_test = dl_cv(X, y_train, model, n_outer, n_inner, compile_kwargs, inner_fit_kwargs, random_state=random_state)

y_pred_df = pd.DataFrame({"predictions": y_pred, "labels": y_test})

y_pred_df.to_csv(os.path.join(save_path, "ERK_Akt.csv"))






