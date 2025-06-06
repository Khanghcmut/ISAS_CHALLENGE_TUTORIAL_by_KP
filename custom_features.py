from tsfel.feature_extraction.features_utils import set_domain
import numpy as np
import pandas as pd
from scipy import stats
@set_domain("domain", "Custom")
def Mode(signal):
    """
    Calculate the mode (most frequent value) of the input signal.

    Parameters:
        signal (array-like): 1D array, list, or Series containing numerical data.

    Returns:
        float: Mode of the signal. If multiple modes, returns the smallest one.
    """
    mode_result = stats.mode(signal, keepdims=False)
    return mode_result.mode
