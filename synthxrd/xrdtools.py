# -*- coding: utf-8 -*-

import imageio
from io import StringIO
import os
import re
import math
import warnings
import logging
from functools import lru_cache, partial
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence, Mapping

import h5py
from scipy import fftpack, ndimage, signal, stats
from silx.io.specfile import SpecFile
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
import xrayutilities as xru
from xml.etree import ElementTree


log = logging.getLogger(__name__)


LAMBDA = 0.43326450378946606 # X-ray wavelength in angstroms
DEFAULT_MASK = 'masks/lab6_1_S002_00000.mask'
DEFAULT_HDF_FILENAME = 'in_situ_calcination_data.h5'


domain_labels = {
    'q': 'q /$A^{-1}$',
    'twotheta': '2θ°',
    'd': 'd /A',
}


def save_refinements_gsas2(csv_filenames, refinement_name, Iss=None, qss=None, patterns=None, overwrite=False):
    """Save all scans to text files for a GSAS-II sequential refinement.
    
    Either *patterns*, or both *Iss*, *qss* must be given.
    
    """
    # Validate the arguments
    if patterns is None:
        patterns = zip(Iss, qss)
    # Prepare a destination directory if necessary
    dirname = f"{refinement_name}_refinements"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    # Iterate through each scan and save the results
    for csv_filename, Is, qs in zip(csv_filenames, Iss, qss):
        # Determine where to save the file (ie. filepath)
        thisfile = os.path.join(dirname, csv_filename)
        # Convert Q to two-theta
        tth = q_to_twotheta(qs, wavelength=LAMBDA)
        # Save the data to the file
        if not os.path.exists(thisfile) or overwrite:
            pattern = pd.Series(Is, index=tth)
            pattern.to_csv(thisfile, header=False)
        else:
            warnings.warn("Refusing to overwrite file %s" % thisfile)


@lru_cache()
def cif_to_powder(ciffile, I0=100, wavelength=None):
    try:
        xu_cif = xru.materials.CIFFile(ciffile)
        xu_crystal = xru.materials.Crystal(name="b-CsCl", lat=xu_cif.SGLattice())
    except:
        log.error("Failed to load cif: %s", ciffile)
        if hasattr(xu_cif, 'close'):
            xu_cif.close()
        raise
    powder = xru.simpack.smaterials.Powder(xu_crystal, 1)
    opts = {}
    if wavelength is not None:
        opts['wl'] = wavelength
    pdiff = xru.simpack.PowderDiffraction(powder, tt_cutoff=80, **opts)
    return pdiff


def plot_cif(ciffile, ax=None, I0=100, color='C0', label=None, domain='q',
             wavelength=None, alpha=0.5, energy=None, *args, **kwargs):
    if energy is not None:
        wavelength = energy_to_wavelength(energy)
    if wavelength is None:
        wavelength = LAMBDA
    powdermodel = cif_to_powder(ciffile, I0, wavelength=wavelength)
    if ax is None:
        ax = plt.gca()
    # Do the plotting
    label_added = False
    for idx, hkl in enumerate(powdermodel.data):
        data = powdermodel.data[hkl]
        ang = data['ang']
        # Decide if we're doing q or two-theta
        q = twotheta_to_q(2*ang, wavelength=wavelength)
        x = convert_q_domain(q, domain=domain, wavelength=wavelength)
        # Plot reflections
        r = data['r']
        if r > 0.01:
            if not label_added:
                _label = label
                label_added = True
            else:
                _label = None
            line = ax.plot([x, x], [0, r*I0], color=color, alpha=alpha, label=_label, *args, **kwargs)
    return line[0]


def rotate_2d_image(data):
    data = np.swapaxes(data, 0, 1)
    data = data[::-1, ::-1]
    return data


def load_data(fpath='NMC_carbonate_Li2CO3_Al2O3tube_1_S006_00000.tif'):
    data = imageio.imread(fpath)
    # Apply a median filter
    data = ndimage.median_filter(input=data, size=5)
    # Rotate data to match the PONI file
    data = rotate_2d_image(data)
    return data


def fit_background(x, y, npt=15, k=5):
    bg_fitter = bayes.BayesianBackground()
    bg = bg_fitter(x, y, npt=npt, k=5)
    return bg


def load_mask(fp=None):
    if fp is None:
        fp = DEFAULT_MASK
    mask = PIL.Image.open(fp)
    mask = np.asarray(mask)
    return mask


def get_mask(data, threshold=None):
    warnings.warn('Deprecated, use *load_mask*')
    assert False, "Out of date, are you sure?"
    if threshold is None:
        print('Fake mask')
        # mask = load_mask('setup/lab6_1_S002_00000.mask')
        mask = np.zeros_like(data)
    else:
        mask = data < threshold
    return mask


def load_integrator(poni_file='images/lab6_1_S002_00000.poni'):
    # ai = pyFAI.load('lab6_s3.poni')
    # ai = pyFAI.load('coin_cell/S7_lab6.poni')
    # ai = pyFAI.load('images/lab6_10-15-2019a.poni')
    ai = pyFAI.load(str(poni_file))
    # # Fix poni calibration
    # poni = 0.0845 # 0.0852
    # ai.set_poni1(0.0858)
    # ai.set_poni2(0.0843)
    return ai


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


def load_metadata(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load metadata previously saved to HDF5 by ``import_metadata``."""
    metadata = pd.read_hdf(hdf_filename, os.path.join(hdf_groupname, 'metadata'))
    return metadata


def import_metadata(flist, hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Import the recorded temperature data, etc. from the .dat files.
    
    Resulting pandas dataframe is saved an HDF5 file and can be
    retrieved with the ``load_metadata`` function..
    
    """
    all_dfs = []
    for fpath in tqdm(flist, desc='Loading metadata'):
        try:
            this_df = pd.read_csv(fpath, sep=r'\s+')
        except FileNotFoundError:
            warnings.warn("Warning, could not open file {}".format(fpath))
            log.warning("Warning, could not open file {}".format(fpath))
            this_df = pd.DataFrame(index=[0])
        else:
            log.debug("Loaded metadata from file: %s", fpath)
        this_df['filename'] = [fpath]
        all_dfs.append(this_df)
    metadata = pd.concat(all_dfs, sort=False)
    metadata = metadata.set_index(pd.to_datetime(metadata['timestamp']))
    # Add a column for elapsed time in minutes
    t0 = np.min(metadata.index)
    metadata['elapsed_time_s'] = (metadata.index - t0).values.astype('float32')/1e9
    metadata.loc[pd.isnull(metadata.index),'elapsed_time_s'] = np.nan
    # Add a column to save the name of the refinement csv file for later
    metadata['refinement_filename'] = ["{}_{:05d}.csv".format(hdf_groupname, i) for i in range(len(metadata))]
    # Save to the HDF5 file
    metadata.to_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'metadata'))


def import_refinements_gsas2(refinement_csv, hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    # Import the refined parameters
    df = pd.read_csv(refinement_csv, index_col=1)
    df.to_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'refinements', 'parameters'))


def load_refinement_params(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load the refined parameters from disk, and merge with the stored metadata."""
    params = pd.read_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'refinements', 'parameters'))
    # Merge by filename
    metadata = load_metadata(hdf_groupname=hdf_groupname, hdf_filename=hdf_filename)
    refined_params = pd.merge(metadata, params, how='right', left_on='refinement_filename', right_on='name')
    return refined_params


@contextmanager
def load_xrd(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load data previously imported by ``import_xrd``.
    
    This functions as a context manager, and yields HDF5
    datasets. Upon exiting the context, the HDF5 file is closed.
    
    Returns
    =======
    qs
      Open HDF5 dataset with scattering lengths (q) in inverse
      angstroms.
    Is
      Open HDF5 dataset with diffraction intensities.
    
    """
    with h5py.File(hdf_filename, mode='r') as fp:
        qs = fp[os.path.join(hdf_groupname, 'scattering_length_q')]
        Is = fp[os.path.join(hdf_groupname, 'integrated_intensity')]
        yield qs, Is


class SpecScan():
    scan_re = re.compile(
        "#S\s+"
        "([0-9]+)\s+"    # Scan number
        "ascan\s+phi\s+"
        "([-0-9]+)\s+"   # Start
        "([-0-9]+)\s+"   # Stop
        "([0-9]+)\s+"    # N_points
        "([0-9]+)"       # Exposure time
    )
    date_re = re.compile("#D\s+(.*)")
    def __init__(self, spec_lines, ):
        self.parse_spec_lines(spec_lines)

    def xml_lines(self, spec_lines):
        """Generate through the UXML lines and return on the first non-xml
        line."""
        for line in spec_lines:
            if line[:5] == '#UXML':
                yield line
            else:
                # Push the non-XML line back onto the stack
                spec_lines.send(line)
                break

    def parse_xml_lines(self, spec_lines):
        xml_string = "\n".join([l[6:].rstrip('"') for l in spec_lines])
        xml_string = f"<specblock>{xml_string}</specblock>"
        block = ElementTree.fromstring(xml_string)
        for group in block:
            if group.attrib['name'] == "ad_file_info":
                fname = [c.text for c in group.findall("dataset") if c.attrib['name'] == "file_name_last_full"][0]
                self.file_path = "/".join(fname.split('/')[-3:])
    
    def generate_lines(self, spec_lines):
        """Creates a generator with the ability to repeat values using
        ``send()``."""
        for line in spec_lines:
            next_line = yield line.strip()
            # Capture a value so it can be pushed back onto the stack
            if next_line:
                yield None # To return from the original send() call
                yield next_line # Will be yielded on the next ``next()``
    
    def parse_spec_lines(self, spec_lines):
        spec_lines = self.generate_lines(spec_lines)
        for line in spec_lines:
            # Scan header line
            scan_match = self.scan_re.match(line)
            if scan_match:
                self.scan_num = scan_match.group(1)
                self.kphi = int((int(scan_match.group(2)) + int(scan_match.group(3))) / 2)
            # Date line
            date_match = self.date_re.match(line)
            if date_match:
                self.timestamp = date_match.group(1)
            # Metadata line
            if line[:3] == '#L ':
                metadata_str = StringIO("\n".join([line[3:], next(spec_lines)]))
                self.metadata = pd.read_csv(metadata_str, sep=r'\s+')
            # embedded xml parsing
            if line[:5] == '#UXML':
                self.parse_xml_lines([line] + list(self.xml_lines(spec_lines)))


def parse_spec_file(spec_file: Path):
    # Prepare regular expressions for parsing the lines
    # Some generic containers to hold the parsed results
    samples = {
    }
    with open(spec_file, mode='r') as fp:
        line = fp.readline()
        while line:
            # Check if this is the start of a scan block
            is_scan_line = line[:2] == "#S"
            if is_scan_line:
                scanlines = [line]
                while "Trajectory scan completed" not in line:
                    # Extract data from the sample line
                    scanlines.append(line)
                    line = fp.readline()
                # Create the scan object
                scanlines.append(line)
                scan = SpecScan(scanlines)
                # Create a new entry for this sample if needed
                if scan.kphi not in samples.keys():
                    samples[scan.kphi] = []
                # Append this scan number to the list of scans
                samples[scan.kphi].append(scan)
            line = fp.readline()
    return samples


class XRDImporter():
    def __init__(self, poni_file, mask_file):
        self.poni_file = poni_file
        self.mask_file = mask_file
    
    def __call__(self, spec_file: Path, sample_names: Mapping={}, overwrite=True):
        """Find and integrate 2D diffraction patterns from in-situ XRD.
        
        If *spec_file* is given, the file list will be populated
        automatically. Either *spec_file* or *flist* is required.
        
        Parameters
        ==========
        spec_file
          Path to the spec file for this experiment.
        
        """
        # Get metadata from spec file
        samples = parse_spec_file(spec_file)
        if samples.keys() != sample_names.keys():
            msg = ("Parameter *sample_names* does not match spec file. "
                   "Provide a mappable with keys matching these kphi values {}".format(list(samples.keys())))
            raise ValueError(msg)
        # Load the data files for each import
        for kphi in samples.keys():
            self.import_2d_xrd([s.file_path for s in samples[kphi]], sample_names[kphi], overwrite=overwrite)
        # Process the metadata into a pandas dataframe
        
    
    def import_2d_xrd(self, flist: Sequence, hdf_groupname,
                      hdf_filename=DEFAULT_HDF_FILENAME,
                      method='integrator', mask=None, threshold=None,
                      overwrite=False):
        results = []
        ai = load_integrator(poni_file=self.poni_file)
        do_integration = partial(self.integrate_data, integrator=ai,
                                 method=method, mask=mask,
                                 threshold=threshold)
        data_list = []
        qs, Is = [], []
        # Check that target dataset doesn't already exist
        with h5py.File(hdf_filename, mode='a') as fp:
            if hdf_groupname in fp:
                if overwrite:
                    del fp[hdf_groupname]
                else:
                    raise RuntimeError("hdf group %s already exists in file %s" % (hdf_groupname, hdf_filename))
        # Integrate the data
        for fpath in tqdm(flist, desc=hdf_groupname):
            try:
                frame = load_data(fpath)
            except FileNotFoundError:
                log.warning("Warning, could not open file %s", fpath)
            new_q, new_I = do_integration(frame)
            qs.append(new_q)
            Is.append(new_I)
        # Save results to HDF5 file
        with h5py.File(hdf_filename, mode='a') as fp:
            fp.create_dataset(os.path.join(hdf_groupname, 'integrated_intensity'), data=Is)
            fp.create_dataset(os.path.join(hdf_groupname, 'scattering_length_q'), data=qs)
        return qs, Is
    
    def integrate_data(self, data, integrator=None, method='integrator',
                       mask=None, threshold=None, qmin=0.4, npt=None,
                       return_cake=False):
        """Mask pixels and produce a 1D spectrum.
        
        If *return_cake=True*, the return value will be (qs, Is, cake,
        qs_cake, chis), otherwise it will be (qs, Is).
        
        Parameters
        ==========
        data
          2D data as np.ndarray.
        integrator
          A pyFAI integrator, generally loaded from a PONI calibration
          file. If omitted, a default integrator will be used.
        method : optional
          Which approach to take for integration. Possible values are
          'integrator' (default), 'median', and 'mean'.
        mask : optional
          A numpy array with the same shape as *data*, and holds a mask to
          use for excluding pixels. If omitted, a default mask will be
          used.
        threshold : optional
          pixel-value threshold for making a mask (deprecated)
        npt : int, optional
          Number of points across for integration, will be used for both
          1d and 2d integration.
        qmin : optional
          Only return intensities above this q-value. Useful for removing
          artifacts from the beamstop, etc.
        return_cake : optional
          If true, return the 2D cake as well as the 1D integrated
          pattern.
        
        Returns
        =======
        qs : 1D ndarray
          Q scattering lengths for the 1D integrated data
        Is : 1D ndarray
          Integrated intensity data
        cake : 2D ndarray, optional
          Caked 2D data.
        qs_cake : 1D ndarray, optional
          Q scattering lengths for the 2D cake's 1st axis.
        chis : 2D ndarray, optional
          Azimuthal angles for the cake's 0th axis.

        """
        if mask is None:
            mask = load_mask(self.mask_file)
        if integrator is None:
            integrator = load_integrator(self.poni_file)
        # Determine default values for the number of points
        if npt is None:
            rms = lambda x: np.sqrt(np.sum(np.power(x, 2)))
            npt_2d = int(rms(data.shape)/2)
            npt_1d = npt_2d * 4
        else:
            npt_2d = npt
            npt_1d = npt
        # Integrate to 2D patterns (uncaking)
        ai_kw = dict(npt_rad=npt_2d, method='python', unit='q_A^-1')
        cake, qs_cake, chis = integrator.integrate2d(data, **ai_kw)
        cake_mask, _, _ = integrator.integrate2d(mask, **ai_kw)
        cake[cake_mask.astype('bool')] = np.nan
        cake[cake==0] = np.nan
        if method == 'median':
            Is = np.nanmedian(cake, axis=0)
            qs = qs_cake
        elif method == 'mean':
            Is = np.nanmean(cake, axis=0)
            qs = qs_cake        
        elif method == 'integrator':
            qs, Is = integrator.integrate1d(data, npt=npt_1d, mask=mask, unit='q_A^-1')
        else:
            raise ValueError("method must be 'integrator', 'median', or 'mean'.")
        # Apply the mask to the cake for display
        # cake = np.ma.array(cake, mask=cake_mask)
        # Apply the minimum q to the data
        Is = Is[qs>=qmin]
        qs = qs[qs>=qmin]
        # Prepare the return values
        if return_cake:
            ret = (qs, Is, cake, qs_cake, chis)
        else:
            ret = (qs, Is)
        return ret


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


def convert_q_domain(q, domain, wavelength=None):
    """Convert q values to d-spacing ('d') or 2θ° ('twotheta')."""
    if wavelength is None:
        wavelength = LAMBDA
    if domain.lower() == 'q':
        x = q
    elif domain.lower() == 'd':
        x = 2 * np.pi / q
    elif domain.lower() in ['twotheta', 'tth']:
        x = q_to_twotheta(q, wavelength=wavelength)
    else:
        raise ValueError(
            f"domain must be one of 'd', 'q', 'twotheta', 'tth'. "
            "Received {domain}".format(domain))
    return x


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
    
    
def plot_insitu_heatmap(qs, Is, metadata, figsize=(8, 8),
                        ciffiles=[], highlighted_scans=(0,),
                        plot_sqrt=False, plot_log=False, domain='q', vmin=None, vmax=None, cmap='viridis'):
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
    highlighted_scans
      Array of scan indices to plot with solid white lines for
      emphasis.
    plot_sqrt
      If true, the image intensity will show the square-root of the
      diffraction signal.
    
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
    temps = metadata['temp']
    # Check that the qs are the same for all scans and equally spaced
    qs_are_consistent = (np.unique(qs, axis=0).shape[0] == 1)
    if not qs_are_consistent:
        warnings.warn('Scattering lengths are not consistent between scans.')
    qs_are_linear = stats.linregress(np.arange(qs.shape[1]), qs[0]).rvalue == 1.0
    if not qs_are_linear:
        warnings.warn('Scattering lengths are not evenly spaced.')
    # Prepare the figure and subplot axes
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0.,
                                    hspace=0., wspace=0.)
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
    tempax.plot(metadata['temp'], times)
    extent = (qs[0,0], qs[0,-1], times[0], times[-1])
    if plot_sqrt:
        scaling_f = np.sqrt
    elif plot_log:
        scaling_f = np.log
    else:
        scaling_f = np.asarray
    xrdimg = xrdax.imshow(scaling_f(Is), origin='bottom', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(mappable=xrdimg, cax=cax, cmap=cmap)
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
    for (label, cifpath), cifax, idx in zip(ciffiles, cifaxs, range(n_ciffs)):
        plot_cif(str(cifpath), ax=cifax, wavelength=LAMBDA, color=f"C{idx}", label=label)
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


def load_temperatures_old(spec_path):
    sf = SpecFile(spec_path)
    all_scans = {}
    for scan_name in sf.keys():
        scan = sf[scan_name]
        data = np.swapaxes(scan.data, 0, 1)    
        df = pd.DataFrame(data=data, columns=scan.labels)
        all_scans[scan_name] = df
    return all_scans


def twotheta_to_q(twotheta, wavelength=None, energy=None):
    # Convert from X-ray energy to X-ray wavelength
    if energy is not None:
        wavelength = energy_to_wavelength(energy)
    # Convert to q if necessary
    if wavelength is None:
        log.debug("No wavelength given, using default: %f", LAMBDA)
        wavelength = LAMBDA
    q = (4*np.pi/wavelength) * np.sin(np.radians(twotheta/2))
    return q


def q_to_twotheta(q, wavelength=LAMBDA):
    radians = np.arcsin(q / 4 / np.pi * wavelength) * 2
    tth = np.degrees(radians)
    return tth


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
