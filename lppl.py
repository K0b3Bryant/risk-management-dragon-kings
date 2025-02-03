import numpy as np
from scipy.optimize import curve_fit

def lppl(t, A, B, tc, m, C, omega, phi):
    return A + B * (tc - t) ** m * (1 + C * np.cos(omega * np.log(tc - t) + phi))

def fit_lppl(time_series):
    t = np.arange(len(time_series))
    p0 = [max(time_series), -1, max(t) + 10, 0.5, 0.1, 6.3, 0]
    try:
        params, _ = curve_fit(lppl, t, time_series, p0=p0, maxfev=LPPL_PARAMS["max_iter"])
        return params
    except Exception as e:
        print(f"LPPL fitting failed: {e}")
        return None
