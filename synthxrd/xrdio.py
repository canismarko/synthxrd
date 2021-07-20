import sys
import os
import re
import logging
from io import StringIO
import imageio
from pathlib import Path
from contextlib import contextmanager
from typing import Sequence, Mapping
from functools import lru_cache, partial

import pyFAI
from scipy import ndimage
import numpy as np
from tqdm.notebook import tqdm
import h5py
import PIL
from xml.etree import ElementTree
import pandas_ods_reader

from .xrdtools import DEFAULT_HDF_FILENAME
from .xrdutils import rotate_2d_image, twotheta_to_q

import pandas as pd


log = logging.getLogger(__name__)


DEFAULT_GSAS_PATH = "~/miniconda3/envs/xrd/GSASII"


def load_mask(fp):
    mask = PIL.Image.open(fp)
    mask = np.asarray(mask)
    return mask


def import_refinements_gsas2_csv(refinement_csv, hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    # Import the refined parameters
    df = pd.read_csv(refinement_csv, index_col=1)
    df.to_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'refinements', 'parameters'))


def import_refinements_gsas2(refinement_gpx, hdf_groupname,
                             hdf_filename=DEFAULT_HDF_FILENAME, gsas_path=DEFAULT_GSAS_PATH):
    # Try and import
    try:
        gsas
    except NameError:
        sys.path.append(str(Path(gsas_path).expanduser()))
        import GSASIIscriptable as gsas
    # Open the GPX project file
    project = gsas.G2Project(refinement_gpx)
    seq = project.seqref()
    enums = ['x', 'yobs', 'yweight', 'ycalc', 'background', 'residual']
    results = {k: [] for k in enums}
    param_rows = []
    filename_re = re.compile('(?:PWDR )?(.*)')
    phases = project.phases()
    histograms = project.histograms()
    for pattern in tqdm(histograms, desc="Importing"):
        # Extract the data
        data_list = pattern.data['data'][1]
        for key in enums:
            results[key].append(data_list[enums.index(key)])
        # Extract the refined parameters
        histname = pattern.data['data'][-1]
        sample_params = pattern.data['Sample Parameters']
        residuals = pattern.data['data'][0].get('Residuals', {})
        params = {
            'name': filename_re.match(pattern.data['data'][-1]).group(1),
            'displacement': sample_params['DisplaceX'][0],
            'pattern_scale': sample_params['Scale'][0],
            'absorption': sample_params['Absorption'][0],
            'wR': residuals.get('wR', np.nan),
        }
        # Get phase-histogram parameters
        for phase in phases:
            pid = phase['pId']
            params.update(_extract_cell_params(seq, pid, pattern.name))
            params.update(_extract_atom_params(seq, pid, pattern.name))
            try:
                hap_values = phase.getHAPvalues(pattern)
            except KeyError:
                continue
            params.update(_extract_hap_sizes(hap_values, pid))
            params.update(_extract_hap_strains(hap_values, pid))
            params.update(_extract_phase_weight(hap_values, phase, pid))
            params[f"{pid}:*:Scale"] = hap_values['Scale'][0]
        # Add overall weight fractions for the phases
        total_weight = np.nansum([v for k, v in params.items() if k[-3:] == 'Wgt'])
        fracs = {f"{k}Frac": v / total_weight for k, v in params.items() if k[-3:] == 'Wgt'}
        params.update(fracs)
        # Add to the list of imported histograms
        param_rows.append(params)
    # Convert the data to numpy arrays
    for key in enums:
        results[key] = np.asarray(results[key])
    # Convert the refined parameters to a pandas dataframe
    metadata = pd.DataFrame.from_dict(param_rows, orient='columns')
    # Save to disk
    metadata.to_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'refinements', 'parameters'))
    with h5py.File(hdf_filename, mode='a') as h5fp:
        base_grp = h5fp[hdf_groupname]
        for key in enums:
            ds_name = os.path.join('refinements', key)
            if ds_name in base_grp:
                del base_grp[ds_name]
            base_grp.create_dataset(ds_name, data=results[key])


def _extract_phase_weight(hap_values, phase, phase_id):
    scale = hap_values['Scale'][0]
    is_used = hap_values['Use']
    mass = phase['General']['Mass']
    new_key = f"{phase_id}:*:Wgt"
    if is_used:
        weight_frac = {
            new_key: scale * mass,
        }
    else:
        weight_frac = {
            new_key: np.nan,
        }
    return weight_frac


def _extract_atom_params(sequential_refinement, phase, histogram):
    try:
        all_params = sequential_refinement.data[histogram]['parmDict']
    except KeyError:
        return {}
    param_fmt = "{id}::{param}"
    esd_fmt = "esd-{id}::{param}"
    new_params = {}
    atom_num = 0
    while True:
        param_names = ["Ax", "Ay", "Az", "Atype", "Afrac", "AUiso"]
        keys = [f"{phase}::{k}:{atom_num}" for k in param_names]
        try:
            new_params.update({k: all_params[k] for k in keys})
        except KeyError:
            break
        else:
            atom_num += 1
    return new_params


def _extract_cell_params(sequential_refinement, phase, histogram):
    try:
        params, esds, unique = sequential_refinement.get_cell_and_esd(phase, histogram)
    except:
        params = [np.nan] * 7
        esds = [0] * 7
        unique = ()
    unique += (6,) # Include the volume as a unique parameter
    param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Vol']
    param_fmt = "{id}::{param}"
    esd_fmt = "esd-{id}::{param}"
    new_params = {}
    for idx in unique:
        # Add the cell parameter itself
        key = param_fmt.format(id=phase, param=param_names[idx])
        new_params[key] = params[idx]
        # Add the cell parameter's ESD
        key = esd_fmt.format(id=phase, param=param_names[idx])
        new_params[key] = params[idx]
    return new_params


def _extract_hap_sizes(hap_values, phase_idx):
    size_params = {}
    size_values = hap_values['Size']
    mode = size_values[0]
    sizes = size_values[1]
    size_params[f"{phase_idx}:*:Size;i"] = sizes[0]
    if mode == 'uniaxial':
        size_params[f"{phase_idx}:*:Size;a"] = sizes[1]
    return size_params


def _extract_hap_strains(hap_values, phase_idx):
    size_params = {}
    size_values = hap_values['Mustrain']
    mode = size_values[0]
    sizes = size_values[1]
    size_params[f"{phase_idx}:*:Mustrain;i"] = sizes[0]
    if mode == 'uniaxial':
        size_params[f"{phase_idx}:*:Mustrain;a"] = sizes[1]
    return size_params


@contextmanager
def load_xrd(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME, background_subtracted=True):
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
        if background_subtracted:
            Is = fp[os.path.join(hdf_groupname, 'integrated_intensity_bg_subtracted')]
        else:
            Is = fp[os.path.join(hdf_groupname, 'integrated_intensity')]
        yield qs, Is


@contextmanager
def load_xrd_2D(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load area XRD data previously imported by ``import_xrd``.
    
    This functions as a context manager, and yields HDF5
    datasets. Upon exiting the context, the HDF5 file is closed.
    
    Returns
    =======
    frames
      Open HDF5 dataset with diffraction intensities
    
    """
    with h5py.File(hdf_filename, mode='r') as fp:
        frames = fp["/".join([hdf_groupname, "intensity_frames"])]
        yield frames


class IOBase():
    @property
    def wavelength(self):
        """Return the calibrated wavelength, in Angstroms."""
        ai = load_integrator(poni_file=self.poni_file)
        return ai.wavelength / 1e-10 # In angstroms
    
    def __init__(self, poni_file: str, hdf_filename=DEFAULT_HDF_FILENAME):
        self.hdf_filename = Path(hdf_filename)
        self.poni_file = Path(poni_file)


class SpecScan():
    file_path = None
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

    def __repr__(self):
        if self.file_path is not None:
            this_repr = f"<SpecScan: {self.file_path}>"
        else:
            this_repr = f"<SpecScan: {self.scan_num}>"
        return this_repr


def parse_spec_file(spec_file: Path, kphi_tol=0.):
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
                log.debug("Found scan line: %s", line)
                scanlines = [line]
                while "Trajectory scan completed" not in line and line:
                    # Extract data from the sample line
                    scanlines.append(line)
                    line = fp.readline()
                # Check if we reached the end of the file
                if line == "":
                    break
                # Create the scan object
                scanlines.append(line)
                scan = SpecScan(scanlines)
                log.debug("Created scan object: %s", scan)
                # Create a new entry for this sample if needed
                matches = [abs(scan.kphi - kphi) <= kphi_tol for kphi in samples.keys()]
                if matches.count(True) > 1:
                    msg = f"kphi {scan.kphi} matches multiple existing entries: {samples.keys()}. "
                    msg += "Consider lowering the value of *kphi_tol*."
                    raise exceptions.AmbiguousKphi(msg)
                elif matches.count(True) == 0:
                    # No existing kphi's are close, so make a new entry
                    samples[scan.kphi] = []
                    sample_kphi = scan.kphi
                else:
                    # Matched one existing kphi value, so use that
                    sample_kphi = list(samples.keys())[matches.index(True)]
                # Append this scan number to the list of scans
                samples[sample_kphi].append(scan)
            line = fp.readline()
    return samples


def load_data(fpath, rotations=1):
    data = imageio.imread(fpath)
    # Apply a median filter
    data = ndimage.median_filter(input=data, size=5)
    # Rotate data to match the PONI file
    for r in range(rotations):
        data = rotate_2d_image(data)
    return data


def load_1d_dioptas(file_name: Path, qmin: float=0):
    """Load an integrated 1D pattern saved by Dioptas.
    
    Parameters
    ==========
    file_name
      Path to the integrated .xy file to import
    qmin
      Minimum scattering vector length that gets included.
    
    Return
    ======
    pattern : dict
      The integrated data and accompanying metadata. 2θ, Q and
      intensity data are in keys "tth", "q" and "I".
    
    """
    pattern = {}
    # Read data via pandas
    data = pd.read_csv(file_name, skiprows=23, names=("tth", "I"), sep="\s+")
    pattern['tth'] = data.tth
    pattern['I'] = data.I
    # Parse calibration data
    with open(file_name, mode="r") as fp:
        calib = [next(fp) for x in range(23)]
    wl_re = re.compile("# Wavelength: ([-0-9.e]+) m")
    distance_re = re.compile("# Distance Sample-beamCenter: ([.0-9]+) mm")
    for ln in calib:
        # Distance
        match = distance_re.match(ln)
        if match:
            pattern['detector_distance_calibrated'] = float(match.group(1))
        # Wavelength
        match = wl_re.match(ln)
        if match:
            wavelength_A = float(match.group(1)) * 1e10
            pattern['wavelength_A'] = wavelength_A
            pattern['q'] = twotheta_to_q(pattern['tth'], wavelength_A)
    if 'q' in pattern.keys():
        is_in_range = pattern['q'] > qmin
        pattern['q'] = pattern['q'][is_in_range]
        pattern['I'] = pattern['I'][is_in_range]
        pattern['tth'] = pattern['tth'][is_in_range]
    return pattern


def load_xrd_molten_salt_17bm(tiff_dir, integrated_dir, metadata_file=None, qmin: float=0, include_calibrant=False):
    """Load the molten salt data sent to 17-BM for XRD analysis.
    
    Parameters
    ==========
    tiff_dir : Path
      Location of the tiff files that will be imported.
    integrated_dir : Path
      Location of the 1D Dioptas integrated .xy files.
    metadata_file : Path
      Location of the excel file holding the sample descriptions.
    qmin : float
      The minimum q value to export.
    include_calibrant : bool
      If true, include the LaB6 calibration samples.
    
    Returns
    =======
    patterns : pd.DataFrame
      A dataframe with all the XRD scans and their accompanying
      metadata.
    
    """
    df = pd.DataFrame()
    for xyfile in tqdm(list(integrated_dir.iterdir()), desc="Importing"):
        if xyfile.suffix == '.xy':
            tiff_metadata_file = tiff_dir/f"{xyfile.stem}.tif.metadata"
            this_row = load_1d_dioptas(xyfile, qmin=qmin)
            this_row['metadata_file'] = tiff_metadata_file
            # Parse metadata
            metadata = parse_metadata_17bm(tiff_metadata_file)
            this_row['sample_name'] = metadata['sample_name']
            this_row['sample_position'] = int(metadata['sample_position'])
            this_row['barcode'] = metadata['barcode']
            this_row['chemical_formula'] = metadata['chemical_formula']
            this_row['chemical_name'] = metadata['chemical_name']
            this_row['detector_distance'] = round(float(metadata['Distance']))
            is_calibrant = this_row['sample_position'] == 0
            if not is_calibrant or include_calibrant:
                df = df.append(this_row, ignore_index=True)
    df['sample_name'] = df['sample_name'].astype(str)
    # Combine this dataframe with the sample descriptions
    if metadata_file is not None:
        metadf = pandas_ods_reader.read_ods(metadata_file, sheet=0)
        metadf['sample_name'] = metadf['sample_name'].astype(str)
        df = df.merge(metadf, on="sample_name", how="left")
    return df


def parse_metadata_17bm(metadata_file: Path):
    """Load a metadata file and return as dictionary."""
    md = {}
    with open(metadata_file, mode='r') as fp:
        for ln in fp:
            try:
                key, val = ln.split("=", 1)
            except ValueError:
                pass
            else:
                md[key] = val.strip()
    return md


class XRDImporter(IOBase):
    """Callable used for importing in-situ XRD scans from a spec file."""
    
    def __init__(self, poni_file: str, mask_file: str, data_dir: str, background_file=None, hdf_filename=DEFAULT_HDF_FILENAME, use_old_filenames=False):
        """Parameters
        ==========
        poni_file
          Relative path to the PONI calibration file prepared using
          Dioptas.
        mask_file
          Path to a .mask file from Dioptas with the mask for removing
          the beam-stop, etc.
        data_dir
          Directory in which to search for scans. Filenames in the
          spec file will be taken as relative to the directory in
          *data_dir*.
        background_file
          Path to an image of the area detector with no sample. This
          will subtracted from all 2D scans before integration.
        hdf_filename
          Path to the HDF file used to store the imported data. If
          omitted, ``xrdtools.DEFAULT_HDF_FILENAME`` will be used.
        use_old_filenames
          If true, refinement filenames will be
          "{name}_{scan_idx}.csv" instead of
          "{name}_{scan_idx}_{scan_num}.csv" for backwards
          compatibility with legacy refinements.
        
        """
        super().__init__(poni_file=poni_file, hdf_filename=hdf_filename)
        self.mask_file = Path(mask_file)
        self.data_dir = data_dir
        if background_file is not None:
            background_file = Path(background_file)
        self.background_file = background_file
        self.use_old_filenames = use_old_filenames
    
    def __call__(self, spec_file: Path, sample_names: Mapping={}, overwrite: bool=True, kphi_tol: int=0):
        """Find and integrate 2D diffraction patterns from in-situ XRD.
        
        If *spec_file* is given, the file list will be populated
        automatically. Either *spec_file* or *flist* is required.
        
        Parameters
        ==========
        spec_file
          Path to the spec file for this experiment.
        sample_names
          A mapping of kphi positions to sample names. The keys to
          this mapping must match the kphi values extracted from the
          spec file, or else an exception will be raised.
        overwrite
          If true, overwrite any previously imported data in the HDF5
          file with these same *sample_names*.
        kphi_tol
          Scans with a khpi within the value to each other will be
          considered part of the same sample.
        
        """
        # Get metadata from spec file
        log.debug("Beginning import from spec file: %s", str(spec_file))
        samples = parse_spec_file(spec_file, kphi_tol=kphi_tol)
        log.debug("Finished parsing spec file.")
        log.info("Found %d samples.", len(samples.keys()))
        all_dfs = []
        if samples.keys() != sample_names.keys():
            summary_list = [f"{k}° ({len(v)})" for k, v in samples.items()]
            msg = ("Parameter *sample_names* does not match spec file. "
                   "Provide a mappable with keys matching these kphi values {}".format(summary_list))
            raise ValueError(msg)
        # Load the data files for each import
        for kphi in samples.keys():
            self.import_2d_xrd([self.data_dir/s.file_path for s in samples[kphi]], sample_names[kphi], overwrite=overwrite)
            # Process the metadata into a pandas dataframe
            metadata = self.merge_metadata(samples[kphi], sample_name=sample_names[kphi])
            metadata.to_hdf(self.hdf_filename, key="/".join([sample_names[kphi], 'metadata']))
    
    def merge_metadata(self, scans, sample_name):
        """Take individual metadata for scans and merge them into a single dataframe."""
        all_dfs = [s.metadata for s in scans]
        metadata = pd.concat(all_dfs, sort=False)
        metadata['timestamp'] = pd.to_datetime([s.timestamp for s in scans])
        metadata['scan_number'] = [s.scan_num for s in scans]
        metadata = metadata.set_index(pd.to_datetime(metadata['timestamp']))
        # Add a column for elapsed time in minutes
        t0 = np.min(metadata.index)
        metadata['elapsed_time_s'] = (metadata.index - t0).values.astype('float32')/1e9
        metadata.loc[pd.isnull(metadata.index),'elapsed_time_s'] = np.nan
        # Add a column to save the name of the refinement csv file for later
        if self.use_old_filenames:
            fmt = "{name}_{idx:05d}.csv"
        else:
            fmt = "{name}_{idx:05d}_{num:05d}.csv"
        refinement_names = [fmt.format(name=sample_name, idx=i, num=int(s.scan_num))
                            for i, s in enumerate(scans)]
        metadata['refinement_filename'] = refinement_names
        return metadata
    
    def import_2d_xrd(self, flist: Sequence, hdf_groupname,
                      method='integrator', mask=None, threshold=None,
                      overwrite=False):
        from dioptas.model.util.BackgroundExtraction import extract_background
        results = []
        ai = load_integrator(poni_file=self.poni_file)
        do_integration = partial(self.integrate_data, integrator=ai,
                                 method=method, mask=mask,
                                 threshold=threshold)
        qs, Is, Is_subtracted = [], [], []
        # Check that target dataset doesn't already exist
        with h5py.File(self.hdf_filename, mode='a') as fp:
            if hdf_groupname in fp:
                if overwrite:
                    del fp[hdf_groupname]
                else:
                    raise RuntimeError("hdf group %s already exists in file %s" % (hdf_groupname, self.hdf_filename))
        # Integrate the data
        frames = []
        for fpath in tqdm(flist, desc=hdf_groupname):
            try:
                frame = load_data(fpath)
            except FileNotFoundError:
                # File is not available, so prepare a dummy file
                log.warning("Warning, could not open file %s", fpath)
                new_q = qs[0]
                new_I = np.zeros_like(Is[0])
            else:
                frames.append(frame)
                new_q, new_I = do_integration(frame)
            qs.append(new_q)
            Is.append(new_I)
            bg = extract_background(new_q, new_I)
            Is_subtracted.append(new_I - bg)
        # Save results to HDF5 file
        with h5py.File(self.hdf_filename, mode='a') as fp:
            fp.create_dataset(os.path.join(hdf_groupname, 'intensity_frames'), data=frames)
            fp.create_dataset(os.path.join(hdf_groupname, 'integrated_intensity'), data=Is)
            fp.create_dataset(os.path.join(hdf_groupname, 'integrated_intensity_bg_subtracted'),
                              data=Is_subtracted)
            fp.create_dataset(os.path.join(hdf_groupname, 'scattering_length_q'), data=qs)
        return qs, Is
    
    @property
    @lru_cache()
    def background_image(self):
        if self.background_file:
            bg_img = load_data(self.background_file)
        else:
            raise exceptions.NoBackgroundFile("Background file not available.")
        return bg_img

    def integrate_tiff(self, tiff_path, return_cake=False, rotations=1):
        """Load a TIFF and return integrated data.
        
        Parameters
        ==========
        data
          2D data as np.ndarray.
        return_cake : optional
          If true, return the 2D cake as well as the 1D integrated
          pattern.
        rotations : optional
          How many 90° rotations to perform on the 2D image before
          unwarping
        
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
        data = load_data(tiff_path, rotations=rotations)
        return self.integrate_data(data, return_cake=return_cake)
    
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
            print(self.poni_file)
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


class GSASExporter(IOBase):
    """Callable used for exporting in-situ XRD scans for GSAS-II refinement."""
    def __call__(self, sample_name, overwrite=True):
        """Save all scans to text files for a GSAS-II sequential refinement.
        
        Either *patterns*, or both *Iss*, *qss* must be given.
        
        """
        # Prepare a destination directory if necessary
        dirname = f"{sample_name}_refinements"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        # Load data
        metadata = load_metadata(sample_name)
        with load_xrd(sample_name, background_subtracted=False) as (qs, Is):
            # Iterate through each scan and save the results
            for csv_filename, Is, qs in tqdm(zip(metadata['refinement_filename'], Is, qs), total=Is.shape[0]):
                # Determine where to save the file (ie. filepath)
                thisfile = os.path.join(dirname, csv_filename)
                # Convert Q to two-theta
                tth = q_to_twotheta(qs, wavelength=self.wavelength)
                # Save the data to the file
                if not os.path.exists(thisfile) or overwrite:
                    pattern = pd.Series(Is, index=tth)
                    pattern.to_csv(thisfile, header=False)
                else:
                    warnings.warn("Refusing to overwrite file %s" % thisfile)


def load_metadata(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load metadata previously saved to HDF5 by ``import_metadata``."""
    metadata = pd.read_hdf(hdf_filename, os.path.join(hdf_groupname, 'metadata'))
    return metadata


def load_fitting(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load fitting data previously saved to HDF5 by."""
    metadata = pd.read_hdf(hdf_filename, '/'.join([hdf_groupname, 'fitting', 'parameters']))
    return metadata


def load_refinement_params(hdf_groupname, hdf_filename=DEFAULT_HDF_FILENAME):
    """Load the refined parameters from disk, and merge with the stored metadata."""
    params = pd.read_hdf(hdf_filename, key=os.path.join(hdf_groupname, 'refinements', 'parameters'))
    # Merge by filename
    metadata = load_metadata(hdf_groupname=hdf_groupname, hdf_filename=hdf_filename)
    refined_params = pd.merge(metadata, params, how='right', left_on='refinement_filename', right_on='name')
    return refined_params


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
