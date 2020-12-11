import numpy as np
import scimap


peak_qranges = {
    '003': (1.0, 1.6),
}


def get_full_qrange(peaks):
    """Get the needed q range to cover all peaks in a set."""
    qmin = min(peak_qranges[hkl][0] for hkl in peaks)
    qmax = max(peak_qranges[hkl][1] for hkl in peaks)
    return (qmin, qmax)


def fit_peaks(qs, Is, peaks=('003',)):
    """
    
    Parameters
    ==========
    qs :
      2D array of scattering lengths in shape (scan, scattering_pos)
    Is :
      2D array of scattering intensities in shape (scan, scattering_pos)
    num_peaks :
      Number of peaks to fit
    qmin :
      Lower bound on the q range used for fitting
    qmax :
      Upper bound on the q range used for fitting
    
    Returns
    =======
      An array of peak fit objects for each scan, one for each peak.
    
    """
    peak_group = scimap.peakfitting.Peak()
    qmin, qmax = get_full_qrange(peaks)
    print(qmin, qmax)
    is_in_bounds = np.logical_and(np.greater_equal(qs, qmin), np.less_equal(qs, qmax))
    # Go through and do the fitting
    peak_group = scimap.peakfitting.Peak()
    peaks = []
    for q, I, ispeak in zip(qs, Is, is_in_bounds):
        peak_group.fit(q[ispeak], I[ispeak])
        peaks.append(peak_group)
    return peaks
