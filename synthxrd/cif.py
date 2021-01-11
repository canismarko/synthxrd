import logging
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import xrayutilities as xru

from .xrdutils import twotheta_to_q, q_to_twotheta, convert_q_domain

log = logging.getLogger(__name__)


@dataclass
class CIF():
    name: str
    path: Path


class AllCifs():
    _cifs = {}
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


cifroot = Path(__file__).resolve().parent.parent / "cif_files"
all_cifs = AllCifs(
    LCO=CIF(name='$LiCoO_2$', path=cifroot/'LiCoO2_CollCode51182.cif'),
    LC=CIF(name='$Li_2CO_3$', path=cifroot/'Li2CO3_CollCode66941.cif'),
    NiO=CIF(name='NiO', path=cifroot/'NiO_ICSD_CollCode9866.cif'),
    NiO_Vacancies=CIF(name='$Ni_{0.75}O$ (disordered, calc)', path=cifroot/'NiO_nickel_vacancies.cif'),
    LH_HYDRATE=CIF(name=r'$LiOH\bullet H_2O$', path=cifroot/'LiOHH2O_CollCode9138.cif'),
    LH=CIF(name=r'LiOH', path=cifroot/'LiOH_CollCode34888.cif'),
    NMC622=CIF(name=r'NMC-622', path=cifroot/'NMC622_ICSD_CollCode159320.cif'),
    Mn2O3=CIF(name=r'$Mn_2O_3$', path=cifroot/'Mn2O3_CollCode187263.cif'),
    Ni3O4=CIF(name=r'$Ni_{0.75}O$ (ordered, MP)', path=cifroot/'Ni3O4_mp-656887_symmetrized.cif'),
    Co3O4=CIF(name=r'$Co_3O_4$ (spinel)', path=cifroot/'Co3O4_CollCode36256.cif'),
    SiO4=CIF(name=r'$Li_4SiO_4$', path=cifroot/'Li4SiO4_ICSD_CollCode25759.cif'),
    WO3=CIF(name=r'$WO_3$', path=cifroot/'WO3.cif'),
    WO3_B=CIF(name=r'$WO_3$', path=cifroot/'WO3_CollCode32001.cif'),
)


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


def plot_cif(ciffile: CIF, ax=None, I0=100, color='C0', label=None, domain='q',
             wavelength=None, alpha=0.5, energy=None, *args, **kwargs):
    """Plot ticks for a CIF file."""
    if energy is not None:
        wavelength = energy_to_wavelength(energy)
    if wavelength is None:
        if domain not in ['q', 'd']:
            raise AttributeError("*wavelength* is required when plotting CIF in two-theta.")
        else:
            wavelength = 1.
    powdermodel = cif_to_powder(ciffile.path, I0, wavelength=wavelength)
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
                _label = label if label is not None else f"{ciffile.name} CIF"
                label_added = True
            else:
                _label = None
            line = ax.plot([x, x], [0, r*I0], color=color, alpha=alpha, label=_label, *args, **kwargs)
    return line[0]
