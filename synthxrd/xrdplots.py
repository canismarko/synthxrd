import warnings

import logging
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .cif import plot_cif
from .xrdutils import twotheta_to_q, q_to_twotheta, convert_q_domain
from .peakfitting import peak_qranges


log = logging.getLogger(__name__)


def add_hkl_label(ax, hkl, x, y, qmin=None, qmax=None):
    # Limit the q-range for this reflection
    if qmin is None and qmax is None:
        qmin, qmax = peak_qranges[hkl]
    in_qrange = np.logical_and(x < qmax, x > qmin)
    _x = x[in_qrange]
    _y = y[in_qrange]
    # Find x, y position of peak
    i_max = np.argmax(_y)
    x_max = _x[i_max]
    y_max = _y[i_max]
    # Add the label
    ax.text(x_max, y_max * 1.015, s=hkl, rotation=90, va="bottom", ha="center")


def plot_insitu_heatmap(qs, Is, metadata, xrd_ax, temp_ax=None,
                        highlighted_scans=(0,),
                        plot_sqrt=True, domain='q', vmin=None, vmax=None):
    """Plot related data for in-situ heating experiments.
    
    Parameters
    ==========
    qs
      Array of scattering lengths
    Is
      Array of integrated diffraction intensities
    metadata
      iterable with scan metadata as a dictionary.
    xrd_ax
      Matplotlib axes to receive the XRD heatmap data
    temp_ax
      Matplotlib axes to receive the temperature data
    highlighted_scans
      Array of scan indices to plot with solid white lines for
      emphasis.
    plot_sqrt
      If true, the image intensity will show the square-root of the
      diffraction signal.
    
    Returns
    =======
    artists
      A lit of artists created.
    
    """
    # Prepare data
    times = metadata['elapsed_time_s'] / 3600
    temps = metadata['temperature']
    # Check that the qs are the same for all scans and equally spaced
    qs_are_consistent = (np.unique(qs, axis=0).shape[0] == 1)
    if not qs_are_consistent:
        warnings.warn('Scattering lengths are not consistent between scans.')
    qs_are_linear = stats.linregress(np.arange(qs.shape[1]), qs[0]).rvalue == 1.0
    if not qs_are_linear:
        warnings.warn('Scattering lengths are not evenly spaced.')
    # Plot the temperature profile
    if temp_ax is not None:
        temp_ax.plot(temps, times)
        all_axs = (xrd_ax, temp_ax)
    else:
        all_axs = (xrd_ax,)
    extent = (qs[0,0], qs[0,-1], times.iloc[0], times.iloc[-1])
    if plot_sqrt:
        scaling_f = np.sqrt
    else:
        scaling_f = np.asarray
    xrdimg = xrd_ax.imshow(scaling_f(Is), origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    # Annotate and format the axes
    xrd_ax.set_facecolor('grey')
    xrd_ax.set_xlabel("|q| /$A^{-1}$")
    xrd_ax.set_xlim(right=5.3)
    xrd_ax.set_ylabel('')
    xrd_ax.set_yticklabels([])
    if temp_ax is not None:
        temp_ax.set_ylabel("Time /h")
        temp_ax.set_xlabel('Temp /°C')
        vline_kw = dict(ls=':', alpha=0.5, zorder=0)
        temp_ax.axvline(30, **vline_kw)
        temp_ax.axvline(500, **vline_kw)
        temp_ax.axvline(900, **vline_kw)
        temp_ax.set_xticks([30, 500, 900])
    # Make all time axes the same
    for ax in all_axs:
        # valid_times = times[times.index != pd.NaT]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pass
        ax.set_ylim(times.min(), times.max())
    # Plot requested CIF files
    for i in highlighted_scans:
        I = Is[i]
        q = qs[i]
        t = times[i]
        T = temps[i]
        xrd_ax.plot(q, (I - I.min()) / (I.max() - I.min()) / 2 + t, color='white')
        xrd_ax.text(x=7.5, y=t + 0.2, s=f'{T:.0f}°C', ha='right', color='white')
    # Make the tick marks in a different domain if requested
    if domain != 'q':
        xrd_ax.set_yticklabels(convert_q_domain(xrd_ax.yticklabels))
    return [xrdimg]


def plot_insitu_heatmap_with_cifs(qs, Is, metadata, figsize=(8, 8),
                                  ciffiles=[], wavelength=None,
                                  highlighted_scans=(),
                                  plot_sqrt=False, plot_log=False,
                                  domain='q', vmin=None, vmax=None,
                                  cmap='viridis', fig=None):
    """Plot related data for in-situ heating experiments.
    
    Parameters
    ==========
    qs
      Array of scattering lengths
    Is
      Array of integrated diffraction intensities
    metadata
      iterable with scan metadata as a dictionary.
    figsize 
      figure size for plotting
    ciffiles
      iterable of paths to .CIF files that will be plotted below the
      XRD scans. Each entry should be a tuple of (label, fpath).
    wavelength
      X-ray wavelength in angstroms.
    highlighted_scans
      Array of scan indices to plot with solid white lines for
      emphasis.
    plot_sqrt
      If true, the image intensity will show the square-root of the
      diffraction signal.
    fig
      A matplotlib Figure instance to receive the new axes.
    
    Returns
    =======
    fig
      The figure receiving the plots.
    ax
      A list of axes that were drawn on, in order (temperature, XRD,
      cif0, cif1, ...)
    
    """
    print("TODO: merge this function with ``plot_insitu_heatmap``")
    # Prepare data
    times = metadata['elapsed_time_s'] / 3600
    temps = metadata['temperature']
    # Check that the qs are the same for all scans and equally spaced
    qs_are_consistent = (np.unique(qs, axis=0).shape[0] == 1)
    if not qs_are_consistent:
        warnings.warn('Scattering lengths are not consistent between scans.')
    qs_are_linear = stats.linregress(np.arange(qs.shape[1]), qs[0]).rvalue == 1.0
    if not qs_are_linear:
        warnings.warn('Scattering lengths are not evenly spaced.')
    # Prepare the figure and subplot axes
    if fig is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=0., h_pad=0.,
    #                                 hspace=0., wspace=0.)
    n_ciffs = len(ciffiles)
    if n_ciffs > 0:
        height_ratios = (9,) + (1,) * n_ciffs
        gs1 = mpl.gridspec.GridSpec(
            1+n_ciffs, 3, figure=fig,
            width_ratios=(1., 7, 0.3), height_ratios=height_ratios,
        )
    else:
        gs1 = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=(1., 7, 0.3))
    tempax = fig.add_subplot(gs1[0,0])
    xrdax = fig.add_subplot(gs1[0,1])
    cax = fig.add_subplot(gs1[0,2])
    cifaxs = np.asarray([fig.add_subplot(gs1[i,1]) for i in range(1, n_ciffs+1)])
    # Plot the temperature profile
    tempax.plot(temps, times)
    extent = (qs[0,0], qs[0,-1], times[0], times[-1])
    if plot_sqrt:
        scaling_f = np.sqrt
    elif plot_log:
        scaling_f = np.log
    else:
        scaling_f = np.asarray
    xrdimg = xrdax.imshow(scaling_f(Is), origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(mappable=xrdimg, cax=cax)
    # Annotate and format the axes
    xrdax.set_facecolor('black')
    xrdax.set_xlabel("|q| /$A^{-1}$")
    xrdax.set_xlim(right=5.3)
    xrdax.xaxis.tick_top()
    xrdax.xaxis.set_label_position('top')
    xrdax.set_ylabel('')
    xrdax.set_yticklabels([])
    tempax.xaxis.tick_top()
    tempax.xaxis.set_label_position('top')
    tempax.set_ylabel("Time /h")
    tempax.set_xlabel('Temp /°C')
    vline_kw = dict(ls=':', alpha=0.5, zorder=0)
    tempax.axvline(30, **vline_kw)
    tempax.axvline(500, **vline_kw)
    tempax.axvline(900, **vline_kw)
    tempax.set_xticks([30, 500, 900])
    # Make all time axes the same
    for ax in (xrdax, tempax):
        # valid_times = times[times.index != pd.NaT]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pass
        ax.set_ylim(times.min(), times.max())
    # Plot requested CIF files
    for ciff, cifax, idx in zip(ciffiles, cifaxs, range(n_ciffs)):
        plot_cif(ciffile=ciff, ax=cifax, wavelength=wavelength, color=f"C{idx}", label=ciff.name)        
    # Format the CIF axes
    for ax in cifaxs:
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.legend(framealpha=0, loc='upper right', handlelength=0)
    # Align all q axes
    for ax in (xrdax, *cifaxs):
        ax.set_xlim(np.min(qs), np.max(qs))
    # Align all the cif vertical axes
    for ax in cifaxs:
        ax.set_ylim(0)
    for i in highlighted_scans:
        I = Is[i]
        q = qs[i]
        t = times[i]
        T = temps[i]
        xrdax.plot(q, (I - I.min()) / (I.max() - I.min()) / 2 + t, color='white')
        xrdax.text(x=7.5, y=t + 0.2, s=f'{T:.0f}°C', ha='right', color='white')
    # Make the tick marks in a different domain if requested
    if domain != 'q':
        xrdax.set_yticklabels(convert_q_domain(xrdax.yticklabels))
    return fig, (tempax, xrdax, *cifaxs)


def plot_insitu_waterfall(qs, Is, metadata, figsize=(8, 8),
                          ciffiles=[], wavelength=None,
                          plot_sqrt=False, plot_log=False, domain='q',
                          cmap='viridis', scale=1, linewidth=0.7, fig=None):
    """Plot related data for in-situ heating experiments.
    
    Parameters
    ==========
    qs
      Array of scattering lengths
    Is
      Array of integrated diffraction intensities
    metadata
      iterable with scan metadata as a dictionary.
    figsize
      figure size for plotting
    ciffiles
      iterable of paths to .CIF files that will be plotted below the
      XRD scans. Each entry should be a tuple of (label, fpath).
    wavelength
      X-ray wavelength in angstroms.
    plot_sqrt
      If true, the image intensity will show the square-root of the
      diffraction signal.
    scale
      Multiplier for scaling the XRD intensity, useful for making the
      plots easier to see
    fig
      Matplotlib Figure that will receive the new axes, etc.
    
    Returns
    =======
    fig
      The figure receiving the plots.
    ax
      A list of axes that were drawn on, in order (temperature, XRD,
      cif0, cif1, ...)
    
    """
    # Prepare data
    times = metadata['elapsed_time_s'] / 3600
    temps = metadata['temperature']
    # Check that the qs are the same for all scans and equally spaced
    qs_are_consistent = (np.unique(qs, axis=0).shape[0] == 1)
    if not qs_are_consistent:
        warnings.warn('Scattering lengths are not consistent between scans.')
    # qs_are_linear = stats.linregress(np.arange(qs.shape[1]), qs[0]).rvalue == 1.0
    # if not qs_are_linear:
    #     warnings.warn('Scattering lengths are not evenly spaced.')
    # Prepare the figure and subplot axes
    if fig is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=0., h_pad=0.,
    #                                 hspace=0., wspace=0.)
    n_ciffs = len(ciffiles)
    if n_ciffs > 0:
        height_ratios = (9,) + (1,) * n_ciffs
        gs1 = mpl.gridspec.GridSpec(
            1+n_ciffs, 2, figure=fig,
            width_ratios=(1., 7.3), height_ratios=height_ratios,
        )
    else:
        gs1 = mpl.gridspec.GridSpec(1, 3, figure=fig, width_ratios=(1., 7, 0.3))
    tempax = fig.add_subplot(gs1[0,0])
    xrdax = fig.add_subplot(gs1[0,1])
    cifaxs = np.asarray([fig.add_subplot(gs1[i,1]) for i in range(1, n_ciffs+1)])
    # Plot the temperature profile
    tempax.plot(temps, times)
    extent = (qs[0,0], qs[0,-1], times[0], times[-1])
    if plot_sqrt:
        scaling_f = np.sqrt
    elif plot_log:
        scaling_f = np.log
    else:
        scaling_f = np.asarray
    # Normalize and plot each XRD pattern
    linespace = np.max(times) / Is.shape[0]
    # Normalize the plot
    I_norm = (Is - np.min(Is)) / (np.max(Is) - np.min(Is))
    I_norm = linespace * scale * I_norm
    # Prepare color mappable
    nrange = (np.min(temps), np.max(temps))
    norm = plt.Normalize(nrange[0], nrange[1] + (nrange[1]-nrange[0])*0.3)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    for idx, (q, I, T) in enumerate(zip(qs, I_norm, temps)):
        xrdax.plot(q, I + linespace * idx, linewidth=linewidth, zorder=Is.shape[0]-idx, color=mappable.to_rgba(T))
    # Annotate and format the axes
    xrdax.set_xlabel("|q| /$A^{-1}$")
    xrdax.set_xlim(right=5.3)
    xrdax.xaxis.tick_top()
    xrdax.xaxis.set_label_position('top')
    xrdax.set_ylabel('')
    xrdax.set_yticklabels([])
    tempax.xaxis.tick_top()
    tempax.xaxis.set_label_position('top')
    tempax.set_ylabel("Time /h")
    tempax.set_xlabel('Temp /°C')
    vline_kw = dict(ls=':', alpha=0.5, zorder=0)
    tempax.axvline(30, **vline_kw)
    tempax.axvline(500, **vline_kw)
    tempax.axvline(900, **vline_kw)
    tempax.set_xticks([30, 500, 900])
    # Make all time axes the same
    for ax in (xrdax, tempax):
        # valid_times = times[times.index != pd.NaT]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pass
        ax.set_ylim(times.min(), times.max())
    # Plot requested CIF files
    for ciff, cifax, idx in zip(ciffiles, cifaxs, range(n_ciffs)):
        plot_cif(ciffile=ciff, ax=cifax, wavelength=wavelength, color=f"C{idx}", label=ciff.name)
    # Format the CIF axes
    for ax in cifaxs:
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.legend(framealpha=0, loc='upper right', handlelength=0)
    # Align all q axes
    for ax in (xrdax, *cifaxs):
        ax.set_xlim(np.min(qs), np.max(qs))
    # Align all the cif vertical axes
    for ax in cifaxs:
        ax.set_ylim(0)
    # Make the tick marks in a different domain if requested
    if domain != 'q':
        xrdax.set_yticklabels(convert_q_domain(xrdax.yticklabels))
    return fig, (tempax, xrdax, *cifaxs)


