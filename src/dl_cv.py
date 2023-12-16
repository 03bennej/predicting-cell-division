import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

random_state=42
n_outer=10
n_inner=10

def model_set_weights(model, compile_kwargs, weights):
    model.compile(**compile_kwargs)
    model.set_weights(weights)
    return model

def get_cv_splits(n_outer, n_inner, random_state=None):
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    return outer_cv, inner_cv

def stack_modalities(modality_1, modality_2):
    modality_1 = np.expand_dims(modality_1, axis=-1)
    modality_2 = np.expand_dims(modality_2, axis=-1)
    return np.concatenate([modality_1, modality_2], axis=-1)

def dl_cv(X, y, model, n_outer, n_inner, compile_kwargs, inner_fit_kwargs, random_state=None):

    print("Beginning cross validation")

    y_pred_combined = []
    y_test_combined = []

    init_weights = model.get_weights()

    sampler = RandomUnderSampler(random_state=42)

    outer_cv, inner_cv = get_cv_splits(n_outer, n_inner, random_state)

    X1 = X[:, :, 0]
    if X.shape[-1] == 2:
         X2 = X[:, :, 1]

    for outer_train_index, outer_test_index in outer_cv.split(X1, y):
        X1_train_outer, X1_test = X1[outer_train_index], X1[outer_test_index]
        y_train_outer, y_test = y[outer_train_index], y[outer_test_index]

        if X.shape[-1] == 2:
            X2_train_outer, X2_test = X2[outer_train_index], X2[outer_test_index]
            X_test = stack_modalities(X1_test, X2_test)
        else:
            X_test = np.expand_dims(X1_test, axis=-1)

        outer_fit_kwargs = {
                "batch_size": inner_fit_kwargs["batch_size"],
                }

        best_epochs = []

        print("Inner cv to determine optimal epoch: \n")
        for inner_train_index, inner_test_index in inner_cv.split(X1_train_outer, y_train_outer):
                model = model_set_weights(model, compile_kwargs, init_weights)
                X1_train_inner, X1_val = X1_train_outer[inner_train_index], X1_train_outer[inner_test_index]
                y_train_inner, y_val = y_train_outer[inner_train_index], y_train_outer[inner_test_index]

                X1_resampled, y_resampled = sampler.fit_resample(X=X1_train_inner, y=y_train_inner)

                if X.shape[-1] == 2:
                    X2_train_inner, X2_val = X2_train_outer[inner_train_index], X2_train_outer[inner_test_index]
                    X2_resampled, _ = sampler.fit_resample(X=X2_train_inner, y=y_train_inner)
                    X_resampled = stack_modalities(X1_resampled, X2_resampled)
                    X_val = stack_modalities(X1_val, X2_val)
                else:
                    X_resampled = np.expand_dims(X1_resampled, axis=-1)
                    X_val = np.expand_dims(X1_val, axis=-1)
                
                history = model.fit(X_resampled, y_resampled, validation_data=(X_val, y_val), **inner_fit_kwargs, verbose=1)
                epoch = np.argmin(history.history["val_loss"])
                val_loss = np.min(history.history["val_loss"])
                best_epochs.append(epoch)
                print(f"Minimum validation loss for epoch {epoch} at {val_loss}")

        epoch_final = int(np.round(np.median(best_epochs)))
        print(f"\n Set epoch to {epoch_final}")
        outer_fit_kwargs["epochs"] = epoch_final


        model = model_set_weights(model, compile_kwargs, init_weights)

        X1_resampled, y_resampled = sampler.fit_resample(X=X1_train_outer, y=y_train_outer)

        if X.shape[-1] == 2:
            X2_resampled, _ = sampler.fit_resample(X=X2_train_outer, y=y_train_outer)
            X_resampled = stack_modalities(X1_resampled, X2_resampled)
        else:
            X_resampled = np.expand_dims(X1_resampled, axis=-1)

        model.fit(X_resampled, y_resampled, **outer_fit_kwargs, verbose=1)
        y_pred = np.squeeze(model.predict(X_test))
        y_pred_combined.append(y_pred)
        y_test_combined.append(y_test)

    y_pred_combined = np.concatenate(y_pred_combined)
    y_test_combined = np.concatenate(y_test_combined)

    return y_pred_combined, y_test_combined