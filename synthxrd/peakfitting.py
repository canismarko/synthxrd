import warnings

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import scimap
from tqdm import tqdm

from . import exceptions


# These q-ranges are specific to layered NMC-type cathodes
peak_qranges = {
    '003': (1.0, 1.6),
    '104': (2.9, 3.25),
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


def fit_peaks(qs: ArrayLike, Is: ArrayLike, dataframe_index, tths: ArrayLike=None, peaks=('003',), method="gaussian", desc="Fitting", centers=None) -> pd.DataFrame:
    """Fit a specific reflection to all the patterns in *qs*, *Is*.
    
    Peaks can also be overlapping adjacent peaks, but this will not
    work well if the peaks are too far separated.
    
    Parameters
    ==========
    qs :
      2D array of scattering lengths in shape (scan, scattering_pos)
    tths
      2D array of scattering angles in shape (scan, scattering_pos)
    Is :
      2D array of scattering intensities in shape (scan, scattering_pos)
    dataframe_index :
      The index of the metadata dataframe. This will be used as the
      index on the output dataframe.
    peaks
      Ranges for the peaks being fit. Can be defined by name in
      ``peak_qranges``, or else a sequence of (qmin, qmax) tuples for
      each peak.
    desc
      Description of the calculation that gets passed to the progress
      bar. Only used if more than one pattern is passed
      (i.e. qs.shape[0] > 1)
    centers
      Initial guesses for the centers of the peaks (in Q).
    
    Returns
    =======
    df : pd.DataFrame
      The fitted parameters and peak objects arranged in a pandas
      DataFrame.

    """
    # if len(peaks) > 1:
    #     raise NotImplementedError("Overlapping peaks not yet implemented.")
    if tths is None:
        tths = [None] * len(qs)
    qmin, qmax = get_full_qrange(peaks)
    # Go through and do the fitting
    peaks_df = pd.DataFrame()
    qs = np.asarray(qs)
    data_bundle = zip(qs, tths, Is, dataframe_index)
    if qs.shape[0] > 1:
        # Wrap in a progress bar
        data_bundle = tqdm(data_bundle, desc=desc, total=qs.shape[0])
    for q, tth, I, idx in data_bundle:
        q = np.asarray(q)
        I = np.asarray(I)
        is_peak = np.logical_and(np.greater_equal(q, qmin), np.less_equal(q, qmax))
        peak_group = scimap.peakfitting.Peak(method=method, num_peaks=len(peaks), centers=centers)
        peak_group.fit(q[is_peak], I[is_peak])
        # Append the fitted row to the pandas DataFrame
        predicted = peak_group.predict(q)
        area = peak_group.area()
        payload = {
            'center_q': peak_group.center(),
            'fwhm_mean': peak_group.fwhm_mean(),
            'fwhm_overall': peak_group.fwhm_overall(),
            'area': area,
            'breadth': area / np.max(predicted),
            'peak': peak_group,
        }
        # Fitting in the 2θ domain
        if tth is not None:
            peak_group = scimap.peakfitting.Peak(method=method)
            peak_group.fit(tth[is_peak], I[is_peak])
            # Append the fitted row to the pandas dataframe
            predicted = peak_group.predict(tth)
            area = peak_group.area()
            payload.update({
                'center_tth': peak_group.center(),
                'fwhm_tth': peak_group.fwhm(),
                'area_tth': area,
                'breadth_tth': area / np.max(predicted),
                'peak_tth': peak_group,
            })
        # Add to the running dataframe
        pd.DataFrame(payload, index=[idx])
        peaks_df = peaks_df.append(other=pd.DataFrame(payload, index=[idx]))
    # Add some additional columns to the dataframe
    peaks_df['center_d'] = 2 * np.pi / peaks_df.center_q
    return peaks_df
