import warnings

import pandas as pd
import numpy as np
import scimap
from tqdm.notebook import tqdm

from . import exceptions


# These q-ranges are specific to layered NMC-type cathodes
peak_qranges = {
    '003': (1.0, 1.6),
    '104': (2.9, 3.2),
    'LaB6-210': (3.325, 3.425),
}


def get_full_qrange(peaks):
    """Get the needed q range to cover all peaks in a set."""
    # Resolve any peaks pre-defined by HKL
    peaks = [(peak_qranges[p] if p in peak_qranges.keys() else p) for p in peaks]
    # Find the min and max q value
    try:
        mins = [min(peak) for peak in peaks]
        maxs = [max(peak) for peak in peaks]
    except KeyError:
        missing_hkl = [hkl for hkl in peaks if hkl not in peak_qranges]
        raise exceptions.UnknownHKL("peak_qranges missing definitions: {}"
                                    "".format(missing_hkl)) from None
    # Check for non-overlapping regions
    minmaxs = list(zip(mins, maxs))
    for (rangeA, rangeB) in zip(minmaxs, minmaxs[1:]):
        has_qgap = rangeA[1] < rangeB[0]
        if has_qgap:
            warnings.warn("Q ranges for peak fitting do not overlap, "
                          "consider fitting distinct peaks one at a time.",
                          RuntimeWarning)
            break
    return (min(mins), max(maxs))


def fit_peaks(qs, Is, peaks=('003',), method="gaussian")->pd.DataFrame:
    """Fit a specific reflection to all the patterns in *qs*, *Is*.
    
    Peaks can also be overlapping adjacent peaks, but this will not
    work well if the peaks are too far separated.
    
    Parameters
    ==========
    qs :
      2D array of scattering lengths in shape (scan, scattering_pos)
    Is :
      2D array of scattering intensities in shape (scan, scattering_pos)
    peaks
      An iterable of hkl indices to fit. Must be defined in ``peak_qranges``.
    
    Returns
    =======
    df : pd.DataFrame
      The fitted parameters and peak objects arranged in a pandas
      DataFrame.
    
    """
    if len(peaks) > 1:
        raise NotImplementedError("Overlapping peaks not yet implemented.")
    qmin, qmax = get_full_qrange(peaks)
    # is_in_bounds = np.logical_and(np.greater_equal(qs, qmin), np.less_equal(qs, qmax))
    # Go through and do the fitting
    peaks = pd.DataFrame()
    for q, I, idx in tqdm(zip(qs, Is, qs.index), desc="Fitting", total=qs.shape[0]):
        q = np.asarray(q)
        I = np.asarray(I)
        is_peak = np.logical_and(np.greater_equal(q, qmin), np.less_equal(q, qmax))
        peak_group = scimap.peakfitting.Peak(method=method)
        peak_group.fit(q[is_peak], I[is_peak])
        # Append the fitted row to the pandas dataframe
        predicted = peak_group.predict(q)
        area = peak_group.area()
        new_row = pd.DataFrame({
            'center_q': peak_group.center(),
            'fwhm': peak_group.fwhm(),
            'area': area,
            'breadth': area / np.max(predicted),
            'peak': peak_group,
        }, index=[idx])
        peaks = peaks.append(other=new_row)
    # Add some additional columns to the dataframe
    peaks['center_d'] = 2 * np.pi / peaks.center_q
    return peaks
