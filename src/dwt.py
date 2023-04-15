import numpy as np
import pandas as pd 
from copy import copy, deepcopy
import pywt

def low_pass_filter_single_signal(signal, thresh = 0.05, wavelet="db3"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="constant" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="constant" )
    return reconstructed_signal

def low_pass_filter(df, thresh, wavelet):
    df_transformed = copy(df)
    for example_id in range(df.shape[0]):
        transformed_example = pd.Series(low_pass_filter_single_signal(df.to_numpy()[example_id, :], thresh, wavelet))
        df_transformed.iloc[example_id, :] = transformed_example

    return df_transformed

def dwt(df, wavelet, mode='constant', level=None, axis=-1):
    
    df = copy(df).to_numpy()
    coeffs = pywt.wavedec(df, wavelet, mode, level, axis)
    dfs = [pd.DataFrame(array) for array in coeffs]

    return dfs


def idwt(coeffs_dfs, wavelet, mode="constant", axis=-1):

    coeffs = [df.to_numpy() for df in coeffs_dfs]

    data = pywt.waverec(coeffs, wavelet, mode, axis)

    return pd.DataFrame(data)

def haar_point_inverse(coeffs, index, mode="constant", axis=-1):

    coeffs = deepcopy(coeffs)
    approximation = deepcopy(coeffs[0].to_numpy())
    approximation[:, np.arange(approximation.shape[1])!=index] = np.nan
    coeffs[0] = pd.DataFrame(approximation)

    return idwt(coeffs, wavelet="haar", mode=mode, axis=axis)

