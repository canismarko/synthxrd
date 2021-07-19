from dataclasses import dataclass

import numpy as np

from synthxrd import q_to_twotheta

@dataclass
class ScherrerModel():
    U: float = 0.
    V: float = 0.
    W: float = 0.
    X: float = 0.
    Y: float = 0.
    Z: float = 0.
    shape_factor: float = 1.
    wavelength: float = 0.45237
        
    def size(self, peak_fwhm: float, peak_twotheta: float)-> float:
        """Calculate the particle size by Sherrer analysis.
        
        Parameters
        ==========
        peak_fwhm
          Full-width at half maximum (in degrees) of the peak.
        peak_twotheta
          Position of the peak, in degrees.
          
        Returns
        =======
        size
            The effective size of coherent scattering domains,
            with units matching ``self.wavelength``.
            
        """
        theta = np.radians(peak_twotheta / 2)
        instrument_fwhm = self.instrument_fwhm(twotheta=peak_twotheta)
        beta = np.radians(peak_fwhm - instrument_fwhm)
        size = self.shape_factor * self.wavelength / beta / np.cos(theta)
        return size
        
    def instrument_fwhm(self, *, twotheta: float=None, q: float=None) -> float:
        """Either *peak_twotheta* or *peak_q* is required."""
        # Convert q to 2θ, if necessary
        if twotheta is None:
            twotheta = synthxrd.q_to_twotheta(q, wavelength=self.wavelength)
        theta = np.radians(twotheta / 2)
        s = np.sqrt(self.U * np.tan(theta)**2 + self.V * np.tan(theta) + self.W)
        g = self.Z + self.X / np.cos(theta) + self.Y * np.tan(theta)
        gamFW = lambda s,g: np.exp(np.log(s**5+2.69269*s**4*g+2.42843*s**3*g**2+4.47163*s**2*g**3+0.07842*s*g**4+g**5)/5.)
        return gamFW(2.35482*s,g) / 100.
