import logging
import re
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xru

from .xrdutils import twotheta_to_q, q_to_twotheta, convert_q_domain

log = logging.getLogger(__name__)


@dataclass
class ReferencePattern():
    def __init__(self, intensity: Sequence, q: Sequence=None, tth: Sequence=None, wavelength=None, name=None):
        """Create a defined reference pattern from individual reflections. 

        Parameters
        ==========
        intensities
          A listing of the relative intensities of each reflection.
        qs
          A listing of the reciprocal scattering lengths for each reflection.
        tths
          A listing of the two-theta values for each reflections.

        """
        if q is None and tth is None:
            raise AttributeError("Either *q* or *tth* is required")
        self.intensity = np.asarray(intensity)
        q = np.asarray(q) if q is not None else q
        tth = np.asarray(tth) if tth is not None else tth
        if q is None:
            if wavelength is None:
                raise AttributeError(
                    "*wavelength* is required when defining pattern in two-theta.")
            q = twotheta_to_q(tth, wavelength=wavelength)
        self.q = q
        self.name = name
        
    def plot(self, ax=None, I0=100, color='C0',
             label=None, domain='q', wavelength=None, alpha=0.5,
             energy=None, *args, **kwargs):
        """Plot ticks for a CIF file."""
        if energy is not None:
            wavelength = energy_to_wavelength(energy)
        if wavelength is None and domain == "tth":
            raise AttributeError(
                    "*wavelength* is required when plotting CIF in two-theta.")
        if ax is None:
            ax = plt.gca()
        # Decide if we're doing q or two-theta
        x = q_to_twotheta(self.q, wavelength=wavelength) if domain == "tth" else self.q
        # Do the plotting
        label_added = False
        for x_, I_ in zip(x, self.intensity):
            # Plot reflections
            if not label_added:
                _label = self.name
                label_added = True
            else:
                _label = None
            line = ax.plot([x_, x_], [0, I_], color=color,
                           alpha=alpha, label=_label, *args, **kwargs)
        return
        


class CIF(ReferencePattern):
    _name: str
    _path: Path
    icsd_re = re.compile("[a-zA-Z0-9_()]+_CollCode([0-9]+).cif")
        
    def __init__(self, name, path):
        self._name = name
        self._path = path

    def __eq__(self, other):
        return (self.name == other.name) and (self.path == other.path)

    def __hash__(self):
        return hash((self.name, self.path))

    @property
    def icsd_code(self):
        match = self.icsd_re.match(self.path.name)
        if match:
            return match.group(1)
        else:
            return None

    @lru_cache()
    def powder_pattern(self, I0=100, wavelength=None):
        try:
            xu_cif = xru.materials.CIFFile(self.path)
            xu_crystal = xru.materials.Crystal(name="b-CsCl", lat=xu_cif.SGLattice())
        except Exception:
            log.error("Failed to load cif: %s", self.path)
            if hasattr(xu_cif, 'close'):
                xu_cif.close()
            raise
        powder = xru.simpack.smaterials.Powder(xu_crystal, 1)
        opts = {}
        if wavelength is not None:
            opts['wl'] = wavelength
        pdiff = xru.simpack.PowderDiffraction(powder, tt_cutoff=80, **opts)
        return pdiff
        
    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    def plot(self, ax=None, I0=100, color='C0',
             label=None, domain='q', wavelength=None, alpha=0.5,
             energy=None, *args, **kwargs):
        """Plot ticks for a CIF file."""
        if energy is not None:
            wavelength = energy_to_wavelength(energy)
        if wavelength is None:
            if domain not in ['q', 'd']:
                raise AttributeError(
                    "*wavelength* is required when plotting CIF in two-theta.")
            else:
                wavelength = 1.
        powdermodel = self.powder_pattern(I0, wavelength=wavelength)
        if ax is None:
            ax = plt.gca()
        # Do the plotting
        label_added = False
        for idx, hkl in enumerate(powdermodel.data):
            data = powdermodel.data[hkl]
            ang = data['ang']
            # Decide if we're doing q or two-theta
            q = twotheta_to_q(2 * ang, wavelength=wavelength)
            x = convert_q_domain(q, domain=domain, wavelength=wavelength)
            # Plot reflections
            r = data['r']
            if r > 0.01:
                if not label_added:
                    _label = label if label is not None else f"{self.name} CIF"
                    label_added = True
                else:
                    _label = None
                line = ax.plot([x, x], [0, r * I0], color=color,
                               alpha=alpha, label=_label, *args, **kwargs)
        return line[0]


class AllCifs():
    _cifs = {}
    _cif_values = None
    def __init__(self, **kwargs):
        self._cifs = kwargs
    
    def __str__(self):
        s = "[\n"
        for key, val in self._cifs.items():
            this_line = "\t{}: ('{}', {})\n".format(key, val.name, val.path)
            s = s + this_line
        s += "]\n"
        return s
    
    def __dir__(self):
        dirs = super().__dir__()
        # Add a DIR entry for each cif file
        dirs.extend(self._cifs.keys())
        return dirs
    
    def __getattr__(self, name):
        return self._cifs[name]

    def __get__(self, name):
        return self._cifs[name]
    
    def __iter__(self):
        self._cif_values = iter(self._cifs.values())
        return self
    
    def __next__(self):
        return next(self._cif_values)
    
    def __len__(self):
        return len(self._cifs)


cifroot = Path(__file__).resolve().parent.parent / "cif_files"
all_cifs = AllCifs(
    LCO=CIF(name='$LiCoO_2$', path=cifroot / 'LiCoO2_CollCode51182.cif'),
    LC=CIF(name='$Li_2CO_3$', path=cifroot / 'Li2CO3_CollCode66941.cif'),
    NiO=CIF(name='NiO', path=cifroot / 'NiO_ICSD_CollCode9866.cif'),
    NiO_Vacancies=CIF(name='$Ni_{0.75}O$ (disordered, calc)', path=cifroot / 'NiO_nickel_vacancies.cif'),
    LH_HYDRATE=CIF(name=r'$LiOH\bullet H_2O$', path=cifroot / 'LiOHH2O_CollCode9138.cif'),
    LH=CIF(name=r'LiOH', path=cifroot / 'LiOH_CollCode34888.cif'),
    NMC622=CIF(name=r'NMC-622', path=cifroot / 'NMC622_ICSD_CollCode159320.cif'),
    NMC811=CIF(name=r'NMC-811', path=cifroot / 'NMC811_ICSD_CollCode8362.cif'),
    NiOOH=CIF(name="NiOOH", path=cifroot/"NiOOH_beta_CollCode165961.cif"),
    NiOH=CIF(name=r'β-Ni(OH)2', path=cifroot / 'NiOH_CollCode169978.cif'),
    MnCO3=CIF(name=r'$MnCO_3$', path=cifroot / 'MnCO3_ICSD_CollCode8433.cif'),
    NiCO3=CIF(name=r'$NiCO_3$', path=cifroot / 'NiCO3_ICSD_CollCode61067.cif'),
    Mn2O3=CIF(name=r'$Mn_2O_3$', path=cifroot / 'Mn2O3_CollCode187263.cif'),
    calcite=CIF(name=r'$CaCO_3$', path=cifroot / "calcite_ICSD_CollCode18164.cif"),
    Ni3O4=CIF(name=r'$Ni_{0.75}O$ (ordered, MP)', path=cifroot / 'Ni3O4_mp-656887_symmetrized.cif'),
    Co3O4=CIF(name=r'$Co_3O_4$ (spinel)', path=cifroot / 'Co3O4_CollCode36256.cif'),
    SiO4=CIF(name=r'$Li_4SiO_4$', path=cifroot / 'Li4SiO4_ICSD_CollCode25759.cif'),
    WO3=CIF(name=r'$WO_3$', path=cifroot / 'WO3.cif'),
    WO3_B=CIF(name=r'$WO_3$', path=cifroot / 'WO3_CollCode32001.cif'),
    CoOH=CIF(name=r'$CoCO_3$', path=cifroot / "CoOH2_ICSD_CollCode257275.cif"),
    CoCO3=CIF(name=r'$CoCO_3$', path=cifroot / "CoCO3_ICSD_CollCode61066.cif"),
    # Lead-acid structures
    Pb=CIF(name=r"Pb", path=cifroot / "Pb_ICSD_CollCode96501.cif"),
    PbSO4=CIF(name=r"$PbSO_4$", path=cifroot / "PbSO4_ICSD_CollCode154273.cif"),
    PbO2_alpha=CIF(name=r"$\alpha-PbO_2$", path=cifroot / "PbO2_alpha_ICSD_CollCode415268.cif"),
    PbO2_beta=CIF(name=r"$\beta-PbO_2$", path=cifroot / "PbO2_beta_ICSD_CollCode8491.cif"),
    PbO=CIF(name=r"PbO", path=cifroot / "PbO_ICSD_CollCode15466.cif"),
    # LLZO solid electrolytes
    LLZO_c=CIF(name=r"LLZO_c", path=cifroot / "LLZO-c_ICSD_CollCode238685.cif"),
    LLZO_t=CIF(name=r"LLZO-t", path=cifroot / "LLZO-t_ICSD_CollCode238686.cif"),
    # Inactive phases
    BN_h = CIF(name=r"BN_h", path=cifroot / "hBN_CollCode240996.cif"),
)


def plot_cif(ciffile, *args, **kwargs):
    """Legacy function for calling CIF().plot(*args, **kwargs)."""
    return ciffile.plot(*args, **kwargs)
