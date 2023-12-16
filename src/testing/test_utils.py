import pandas as pd
from copy import deepcopy

def apply_threshold(y_pred, threshold):
    y_pred_label = deepcopy(y_pred)
    y_pred_label[y_pred_label>=threshold] = 1
    y_pred_label[y_pred_label<threshold] = 0
    return y_pred_label

def make_predictions_df(EI, X_dict, model_id):
    if EI.meta_models is not None:
        predictions = EI.predict(X_dict, model_id)
        threshold_f = EI.meta_summary["thresholds"][modelID_dict[modality_id]]["fmax (minority)"]

    else:
        all_base_models = deepcopy(EI.final_models["base models"][modality_id])
        base_models = [dictionary for dictionary in all_base_models if dictionary["model name"] == modelID_dict[modality_id]]
        unpickled_base_models = [pickle.loads(base_model["pickled model"]) for base_model in base_models]
        predictions = [model.predict(data_dict[modality_id][modality_id]) for model in unpickled_base_models]
        predictions = np.mean(np.array(predictions), axis=0)
        threshold_f = EI.base_summary["thresholds"].T["fmax (minority)"].to_numpy()

    inference = apply_threshold(predictions, threshold_f)
    df = pd.DataFrame({"inference": inference, "predictions": predictions, "labels": y})
    df.to_csv(os.path.join(save_path, f"predictions_{save_filenames_dict[modality_id]}"))

