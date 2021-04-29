# -*- coding: utf-8 -*-

import imageio
from io import StringIO
import os
import re
import math
import warnings
import logging
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
from scipy import fftpack, ndimage, signal, stats
# from silx.io.specfile import SpecFile
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import skimage
from skimage import morphology
import pyFAI
from pyFAI.utils import bayes
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xanespy as xp
import PIL
from xml.etree import ElementTree
from dioptas.model.util.BackgroundExtraction import extract_background


from . import exceptions


log = logging.getLogger(__name__)


LAMBDA = 0.43326450378946606 # X-ray wavelength in angstroms
DEFAULT_MASK = 'masks/lab6_1_S002_00000.mask'
DEFAULT_HDF_FILENAME = 'in_situ_calcination_data.h5'


domain_labels = {
    'q': 'q /$A^{-1}$',
    'twotheta': '2θ°',
    'd': 'd /A',
}


def fit_background(x, y, npt=15, k=5):
    bg_fitter = bayes.BayesianBackground()
    bg = bg_fitter(x, y, npt=npt, k=5)
    return bg


def get_mask(data, threshold=None):
    warnings.warn('Deprecated, use *load_mask*')
    assert False, "Out of date, are you sure?"
    if threshold is None:
        # mask = load_mask('setup/lab6_1_S002_00000.mask')
        mask = np.zeros_like(data)
    else:
        mask = data < threshold
    return mask


def export_xye(filepath, Is, qs=None, twotheta=None, Es=None):
    # Check that inputs are sane
    valid_x = qs is not None or twotheta is not None
    if not valid_x:
        raise ValueError("Either *qs* or *twotheta* is required.")
    # Prepare the dataframe
    data = {'counts': Is}
    if Es is not None:
        data['error'] = Es
    if twotheta is not None:
        data['2θ°'] = twotheta
    if qs is not None:
        data['Q'] = qs
    df = pd.DataFrame.from_dict(data)
    # Determine what to use as the index
    if twotheta is not None:
        df = df.set_index('2θ°')
    else:
        df = df.set_index('Q')
    # Save to disk
    df.to_csv(filepath)


# def import_metadata(flist, hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
#     """Import the recorded temperature data, etc. from the .dat files.
#     
#     Resulting pandas dataframe is saved an HDF5 file and can be
#     retrieved with the ``load_metadata`` function..
#     
#     """
#     all_dfs = []
#     for fpath in tqdm(flist, desc='Loading metadata'):
#         try:
#             this_df = pd.read_csv(fpath, sep=r'\s+')
#         except FileNotFoundError:
#             warnings.warn("Warning, could not open file {}".format(fpath))
#             log.warning("Warning, could not open file {}".format(fpath))
#             this_df = pd.DataFrame(index=[0])
#         else:
#             log.debug("Loaded metadata from file: %s", fpath)
#         this_df['filename'] = [fpath]
#         all_dfs.append(this_df)
#     metadata = pd.concat(all_dfs, sort=False)
#     metadata = metadata.set_index(pd.to_datetime(metadata['timestamp']))
#     # Add a column for elapsed time in minutes
#     t0 = np.min(metadata.index)
#     metadata['elapsed_time_s'] = (metadata.index - t0).values.astype('float32')/1e9
#     metadata.loc[pd.isnull(metadata.index),'elapsed_time_s'] = np.nan
#     # Add a column to save the name of the refinement csv file for later
#     metadata['refinement_filename'] = ["{}_{:05d}.csv".format(hdf_groupname, i) for i in range(len(metadata))]
#     # Save to the HDF5 file
#     metadata.to_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'metadata'))


# Build the file list
def find_files(directory, curr_list=None):
    if curr_list is None:
        curr_list = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isdir(fpath):
            find_files(fpath, curr_list=curr_list)
        elif os.path.splitext(fpath)[1] == '.tif':
            curr_list.append(fpath)
    return tuple(curr_list)


def plot_scan_summary(filepath, mask_path=None, bg_npt=11, figsize=(15, 4), domain='q'):
    """Plot several views of data integration."""
    fig, (imax, cakeax, xrdax) = plt.subplots(1, 3, figsize=figsize)
    lab6 = load_data(filepath)
    mask = load_mask(mask_path)
    
    extent = (0, 0.172*lab6.shape[1], 0, 0.172*lab6.shape[0])
    imax.imshow(lab6, extent=extent)
    imax.set_xlabel('mm')
    imax.set_ylabel('mm')
    integrator = load_integrator()
   
    ai_kw = dict(npt_rad=500, method='cython')
    qs, Is, cake, qs_cake, chis = integrate_data(lab6, integrator=integrator, mask=mask, return_cake=True)
    extent = (qs_cake.min(), qs_cake.max(), chis.max(), chis.min())
    cakeax.imshow(np.isnan(cake), extent=extent, aspect='auto')
    cakeax.set_xlabel('q /$A^{-1}$')
    cakeax.set_ylabel(r'$\chi ^{\circ}$')
    
    # Fit background
    xs = convert_q_domain(qs, domain)
    xrdax.plot(xs, Is)
    bg = fit_background(xs, Is, npt=bg_npt, k=3)
    xrdax.plot(xs, bg)
    xrdax.plot(xs, Is-bg)
    xrdax.set_xlabel(domain_labels[domain])
    xrdax.set_ylabel("Mean counts")
    
    
def plot_param_with_esd(x, df, col, ax, esd_alpha=0.3, *args, **kwargs):
    param = df[col]
    esd = df[f'esd-{col}']
    ax.plot(x, param, *args, **kwargs)
    # Adjust kwargs for for the esd
    esd_kw = kwargs.copy()
    esd_kw.pop('alpha', None)
    esd_kw.pop('label', None)
    ax.fill_between(x, param-esd, param+esd, alpha=esd_alpha, *args, **esd_kw)
    

def plot_all_files(directory: str, method: str='integrator', domain:
                   str="q", plot_as: str='curves', wavelength:
                   float=None, threshold: float=None,
                   which_patterns=None, temperatures=None):

    """Plot all the files in a given directory as a series of XRD patterns.
    
    Parameters
    ==========
    directory
      Path to the directory that should be search for TIFF XRD
      patterns.
    method
      Describes how to convert 2D patterns to 1D patterns. Valid
      options are 'integrator' (default) or 'median'.
    domain
      Which x-values to use for plotting: 'q' (default), 'd',
      'twotheta'.
    plot_as
      Controls the style of plotting for the full spectra. Options are
      'curves' (default), and 'heatmap'
    wavelength
      The X-ray wavelength, in nm. Needed if domain is 'twotheta'.
    threshold
      Pixel value threshold for binarizing when calculating mask.
    which_patterns
      Slice to use for extracting XRD patterns. Special values of
      'mean' and 'median' can also be given. By default, 10 equally
      spaced patterns will be shown.
    temperatures
      If given, should be a pandas series with the time-temperature
      series to plot.
    
    """
    flist = find_files(directory=directory)
    # Determine which patterns to show
    if which_patterns is None:
        step = int(len(flist) / 10)
        which_patterns = slice(None, None, max(step, 1))
        which_patterns = slice(None, None, None)
    flist = flist[which_patterns]
    # Import the files and integrate
    cakes = []
    qss = []
    Iss = []
    # Prepare plotting area
    if temperatures is None:
        # Only 2 axes for plotting
        fig, (ax_xrd, ax_cake) = plt.subplots(1, 2, figsize=(12, max(len(flist) * 6 / 200, 6)))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, max(len(flist) * 6 / 200, 6)))
        (ax_temp, ax_xrd, ax_cake) = axs
    # Plot the temperature data
    if temperatures is not None:
        temp_x = temperatures.index
        ax_temp.plot(temperatures, temp_x)
        ax_temp.set_ylim(0, np.max(temp_x))
    # Plot each spectrum
    curr_y = 0
    integrated_files = integrate_files(tuple(flist),
                                       method=method,
                                       threshold=threshold)
    for idx, (qs, Is, cake) in enumerate(integrated_files):
        xs = convert_q_domain(qs, domain=domain, wavelength=wavelength)
        if plot_as == 'curves':
            ax_xrd.plot(xs, Is+curr_y)
        # Adjust the vertical offset for the next plot
        curr_y += 0.15 * (np.max(Is) - np.min(Is))
        # Save data for later
        cakes.append(cake)
        qss.append(xs)
        Iss.append(Is)
    qss = np.asarray(qss)
    Iss = np.asarray(Iss)
    # Plot the average cake file
    mean_cake = np.nanmean(cakes, axis=0)
    mean_qss = np.mean(qss, axis=0)
    extent = (np.min(mean_qss), np.max(mean_qss), 0, mean_cake.shape[0]-1)
    ax_cake.imshow(mean_cake, extent=extent, aspect='auto')
    # Format the axes
    qlim = np.array([10, 48])
    qlim = np.array([2, 81])
    if plot_as == 'heatmap':
        # Plot the XRD patterns as a heatmap
        heatmap = ax_xrd.imshow(Iss, extent=(np.min(qss), np.max(qss), 0, Iss.shape[0]), aspect='auto', origin='lower')
        # Now add image formatting
        plt.colorbar(ax=ax_xrd, mappable=heatmap)
    elif plot_as == 'curves':
        xlim = convert_q_domain(q=qlim, domain=domain, wavelength=wavelength)
        ax_xrd.set_xlim(*xlim)
    xlabels = {
        'q': 'q / nm^{-1}',
        'twotheta': '2θ° (λ={} nm)'.format(wavelength),
        'd': 'd / nm'
    }
    xlabel = xlabels[domain.lower()]
    # ax_summ.set_xlim(*xlim)
    ax_xrd.set_xlabel(xlabel)
    ax_cake.set_xlabel(xlabel)
    ax_xrd.set_title("All patterns")
    ax_cake.set_title('Mean cake')
    fig.suptitle(f'{directory} - {method}')
    # Return imported data for further analysis
    return (qss, Iss, cakes)


def plot_temperatures(spec_path):
    data = load_temperatures_old(spec_path)
    # Determine how many axes we need
    n_scans = len(data)
    n_cols = 3
    n_rows = math.ceil(n_scans / n_cols)

    # Create the axes and figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), constrained_layout=True)
    n_leftover_ax = n_scans - n_rows * n_cols
    if n_leftover_ax:
        for ax in axs.flatten()[n_leftover_ax:]:
            ax.remove()
    # Plot each dataset
    artists = []
    for ax, (scan_title, df) in zip(axs.flatten(), data.items()):
        try:
            x = df['Time'].values
        except KeyError:
            x = df.index
            ax.set_xlabel('Index')
        except AssertionError:
            pass
        else:
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature")
        finally:
            artist = ax.plot(x, df['temperature'].values)
            artists.append(artist[0])
            ax.set_title(f'{scan_title} (N={len(df)})')
    return artists


# def load_temperatures_old(spec_path):
#     sf = SpecFile(spec_path)
#     all_scans = {}
#     for scan_name in sf.keys():
#         scan = sf[scan_name]
#         data = np.swapaxes(scan.data, 0, 1)    
#         df = pd.DataFrame(data=data, columns=scan.labels)
#         all_scans[scan_name] = df
#     return all_scans


class XRDPeak(xp.fitting.Curve):
    params_names = ('height', 'center', 'width', 'voffset')
    def __call__(self, height, center, width, voffset):
        gauss = xp.fitting.Gaussian(x=self.x)
        return gauss(height, center, width) + voffset


class RockSaltPeaks(xp.fitting.Curve):
    params_names = ('height0', 'center0', 'width0', 'height1', 'center1', 'width1', 'voffset')
    def __call__(self, height0, center0, width0, height1, center1, width1, height2, center2, width2, voffset):
        gauss = xp.fitting.Gaussian(x=self.x)
        return gauss(height0, center0, width0) + gauss(height1, center1, width1) + gauss(height2, center2, width2) + voffset


def fit_layered(qs, Is, ax=None):
    """Fit the layered peaks and return d-spacing and integral breadth."""
    is001 = np.logical_and(qs>10, qs<15)
    Is001 = Is[is001]
    qs001 = qs[is001]
    # Do the fitting
    peak001 = XRDPeak(qs001)
    v0 = np.min(Is001)
    I0 = np.max(Is001) - v0
    bounds = (
        [0,      0,      0.1,    0,      ],
        [np.inf, np.inf, np.inf, np.inf, ],
    )
    fit, pcov = xp.fitting.fit_spectra([Is001], func=peak001,
                                       p0=[(I0, 13, 0.1, v0)], ncore=1,
                                       quiet=True, bounds=bounds)
    # Plot the resulting fit
    if ax is not None:
        qsfit = np.linspace(np.min(qs001), np.max(qs001), num=2000)
        peakfit = XRDPeak(qsfit)
        ax.plot(qs001, Is001, marker='+', linestyle="None")
        ax.plot(qsfit, peakfit(*fit[0]))
        ax.set_ylim(70, 220)
    # Calculate target values from fitting parameters
    d = 2 * np.pi / fit[0,1]
    intensity = fit[0,0]
    breadth = fit[0, 2] * np.sqrt(2*np.pi)
    return intensity, d, breadth


def fit_rock_salt(qs, Is, ax=None):
    """Fit the rock-salt peaks and return d-spacing and integral breadth."""
    is220 = np.logical_and(qs>40, qs<43.5)
    Is220 = Is[is220]
    qs220 = qs[is220]
    # Do the fitting
    peak220 = RockSaltPeaks(qs220)
    v0 = np.min(Is220)
    c0 = qs220[np.argmax(Is220)]
    I0 = np.max(Is220)
    bounds = [
        (     0,  40.2, -np.inf,      0, 41.1, -np.inf,      0, -np.inf, -np.inf, -np.inf),
        (np.inf, 40.45,  np.inf, np.inf, 41.3,  np.inf, np.inf,  np.inf,  np.inf,  np.inf),
    ]
    p0 = [I0 - v0, 40.3, 0.1, I0 - v0, 41.2, 0.1, I0-v0, c0, 0.1, v0]
    fit, pcov = xp.fitting.fit_spectra([Is220], func=peak220, p0=p0, ncore=1, quiet=True, nonnegative=True, bounds=bounds)
    if xp.fitting.is_out_of_bounds(fit[0], bounds):
        warnings.warn('Fit params out of bounds: {}'.format(fit))
    # Plot the resulting fit
    qsfit = np.linspace(np.min(qs220), np.max(qs220), num=2000)
    peakfit = RockSaltPeaks(qsfit)
    if ax is not None:
        ax.plot(qs220, Is220, marker='+', linestyle="None")
        ax.plot(qsfit, peakfit(*fit[0]))
        gauss = xp.fitting.Gaussian(qsfit)
        line_kw = dict(linestyle=":")
        ax.plot(qsfit, gauss(*fit[0, 0:3]) + fit[0, -1], **line_kw)
        ax.plot(qsfit, gauss(*fit[0, 3:6]) + fit[0, -1], **line_kw)
        ax.plot(qsfit, gauss(*fit[0, 6:9]) + fit[0, -1], **line_kw)
        ax.set_ylim(70, 350)
    # Calculate target values from fitting parameters
    max_phase = np.argmax(fit[0,1::])
    fit_p = fit[0,3*2:3*2+3]
    d = 2 * np.pi / fit_p[1]
    intensity = fit_p[0]
    breadth = fit_p[2] * np.sqrt(2*np.pi)
    return intensity, d, breadth


def plot_xrd_params(directory: str, specfile: str, specscan: str,
                    which_files=slice(None), relative=False, smooth=False,
                    plot_fits=False):
    """Plot changes in XRD parameters for pre-defined phases.

    Parameters
    ==========
    directory
      Path to a directory within which to search for XRD patterns.
    specfile
      Path to a specfile with scan data.
    specscan
      Key for scan in the specfile to use for plotting.
    which_files
      Index for which XRD patterns to process. Default is to use all
      files.
    relative
      If true, values will be plotted relative to the full range of
      values, otherwise absolute values will be plotted.
    smooth
      If true, a mild Savitzky-Golay filter will be applied to the
      time-series data.
    plot_fits
      If true, the fit spectra will also be plotted, one figure for
      each pattern.
    
    """
    raise DeprecationWarning("I think this is an old routine. If needed, it will need to be updated. -Mark")
    # Load image files
    files = find_files(directory)[which_files]
    imgdata = integrate_files(files, method='integrator')
    # Load the spec file data
    sf = SpecFile(specfile)
    scan = sf[specscan]
    data = np.swapaxes(scan.data, 0, 1)
    df = pd.DataFrame(data=data, columns=scan.labels)
    try:
        times = df['Time'][which_files]
    except KeyError:
        use_time_as_x = False
    else:
        if len(files) != len(times):
            use_time_as_x = False
        else:
            use_time_as_x = True
    temperature = df['temperature'][which_files]
    # Create arrays for holding results
    heights = []
    ds = []
    betas = []
    # Do the fitting for each phase
    for qs, Is, cakes in tqdm(imgdata, desc="Fitting"):
        if plot_fits:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
        else:
            ax0, ax1 = (None, None)
        h_lay, d_lay, beta_lay = fit_layered(qs, Is, ax=ax0)
        h_rs, d_rs, beta_rs = fit_rock_salt(qs, Is, ax=ax1)
        heights.append((h_lay, h_rs))
        ds.append((d_lay, d_rs))
        betas.append((beta_lay, beta_rs))
    # Restructure data to be in (phase, param, scan) order
    heights = np.asarray(heights)
    ds = np.asarray(ds)
    betas = np.asarray(betas)
    heights = np.moveaxis(heights, 1, 0)
    ds = np.moveaxis(ds, 1, 0)
    betas = np.moveaxis(betas, 1, 0)
    # Plot the results
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True, figsize=(4, 11), constrained_layout=True)
    for h, d, beta in zip(heights, ds, betas):
        if smooth:
            savgol_kw = dict(window_length=11, polyorder=3)
            h = signal.savgol_filter(h, **savgol_kw)
            d = signal.savgol_filter(d, **savgol_kw)
            beta = signal.savgol_filter(beta, **savgol_kw)
        if relative:
            normalize = lambda x, y: (x-np.min(x)) / (np.max(x) - np.min(x))
            h = normalize(h, heights)
            d = normalize(d, ds)
            beta = normalize(beta, betas)
        if use_time_as_x:
            x = times
            xlabel = "Time"
        else:
            x = range(len(h))
            xlabel = "Index"
        ax0.plot(x, h)
        ax1.plot(x, d)
        ax2.plot(x, beta)
    if use_time_as_x:
        ax3.plot(times, temperature)
    else:
        ax3.plot(temperature)
    # Set default plot limits
    if not relative:
        ax0.set_ylim(bottom=-10)
    # Format the axes 
    ax0.set_title("Peak Intensity")
    ax1.set_title('Peak Position')
    ax2.set_title("Peak Shape")
    ax3.set_title("Temperature")
    if relative:
        ax0.set_ylabel("Relative peak height")
        ax1.set_ylabel("Relative d-spacing")
        ax2.set_ylabel("Relative integral breadth")
    else:
        ax0.set_ylabel("Peak height")
        ax1.set_ylabel("D-spacing /nm")
        ax2.set_ylabel("Integral Breadth")
        ax3.set_ylabel("Temperature /°C")
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlabel(xlabel)
    for ax in (ax0, ax1, ax2):
        ax.legend(['Layered', 'Rock-salt'])
