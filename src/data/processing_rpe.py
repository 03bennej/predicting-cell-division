import numpy as np
import os
import pandas as pd
from pathlib import Path
from src.data.processing_utils import df_to_csv, random_truncate, combine_divided_undivided, mean_pad

train_data_path = "data/processed/mcf10a/high_dose/train"
data_path = "data/raw/rpe/nodox_62point5nm"
save_path = "data/processed/rpe"
params_path = "data/processed/mcf10a/params"

#  Load training
ERK_train_df = pd.read_csv(os.path.join(train_data_path, "ERK.csv"))
ERK_train = ERK_train_df.drop('label', axis=1).to_numpy()

median_train = np.median(ERK_train[:, 6:], axis=0)  # take median after baseline measurements since baseline not included in rpe experiments

#  Load RPE data
ERK = pd.read_csv(os.path.join(data_path, "ERK.csv")).to_numpy()
divisions = pd.read_csv(os.path.join(data_path, "divisions.csv")).to_numpy()

train_lengths = pd.read_csv(os.path.join(params_path, "length_distribution.csv")).to_numpy(dtype=int).flatten()

# Calculate scaling factor and recentering constant from 

#  Begin preprocessing steps

max_timepoints = int(49*60/10 + 1) 
ERK = ERK[:, -max_timepoints:]  # we took last 49 hours of data. For ERKi (62.5nM) this gives a balanced test set
divisions = divisions[:, -max_timepoints:]

ERK_new = np.empty_like(ERK)

division_args = []  # list of arrays of length number of examples. Each array contains division time points, or is empty indicating no division occured
for time_course in divisions:
    division_args.append(np.where(time_course == 1)[0])

division_timepoints = []  # list of floats of length number of examples. Each float is an integer indicating the division time point, or nan indicating no division
for array in division_args:
    if len(array)<1:
        division_timepoints.append(np.nan)
    else:
        division_timepoints.append(array[0])

for id, time_course in enumerate(ERK):  # truncate at division and add nans to right hand side of time course
    division_timepoint = division_timepoints[id]
    if np.isnan(division_timepoint):
        ERK_new[id, :] = ERK[id, :] 
    else:
        ERK_new[id, :] = np.concatenate([ERK[id, :division_timepoint], np.nan * np.ones(max_timepoints - division_timepoint)]) 

labels = np.ones(shape=len(division_timepoints), dtype=int)
labels[np.isnan(division_timepoints)] = 0

t = np.linspace(0, 49*60, max_timepoints)
t_new = np.linspace(0, 49*60, 198)

ERK_interp = []
for time_course in ERK_new:
    time_course_new = np.interp(t_new, t, time_course)
    ERK_interp.append(time_course_new)
ERK_interp = pd.DataFrame(np.stack(ERK_interp))

divided_ERK = ERK_interp[labels==1].to_numpy()
undivided_ERK = ERK_interp[labels==0].to_numpy()

undivided_ERK_trunc, _, _ = random_truncate(divided_ERK, undivided_ERK, length_distribution=train_lengths, random_seed=42)

ERK_final = combine_divided_undivided(divided_ERK, undivided_ERK_trunc)

ERK_final_pad = mean_pad(ERK_final).dropna(axis=0)  # drop series that are all NaNs, i.e. division occurred at time 0.

path_test = os.path.join(save_path, "test")
Path(path_test).mkdir(parents=True, exist_ok=True)

train_median_middle_range = (median_train.min() + median_train.max()) / 2
chen_median_middle_range = (ERK_final_pad.iloc[:, :-1].median().min() + ERK_final_pad.iloc[:, :-1].median().max())/ 2

ERK_final_pad.iloc[:, :-1] = ERK_final_pad.iloc[:, :-1] - chen_median_middle_range + train_median_middle_range

# ERK_final_pad.to_csv(os.path.join(path_test, "ERK.csv"), index=False)
df_to_csv(ERK_final_pad, path_test, 'ERK.csv')