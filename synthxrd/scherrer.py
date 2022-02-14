from dataclasses import dataclass

import numpy as np

from synthxrd import q_to_twotheta, twotheta_to_q


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

    def convert_domain(self, val, src="q", dest="tth"):
        """Convert from one domain to another."""
        # Check for no-ops
        if src == dest:
            return val
        # Do the conversion
        if src == "q" and dest == "tth":
            new_val = q_to_twotheta(val, wavelength=self.wavelength)
        elif src == "tth" and dest == "q":
            new_val = twotheta_to_q(val, wavelength=self.wavelength)
        else:
            raise ValueError('Cannot convert from "{}" to "{}"'.format(src, dest))
        return new_val

    def convert_fwhm_domain(self, fwhm: float, peak_position: float,
                            src="q", dest="tth"):
        """Converts a full-width half max from one domain to another.
        
        E.g. Convert FWHM from 2θ° to q.

        Assumes the peak is roughly symmetric around the peak position.
        
        """
        # Check for no-ops
        if src == dest:
            return fwhm
        # Split up the range into upper and lower bounds
        top = peak_position + fwhm / 2
        btm = peak_position - fwhm / 2
        # Convert each bound from one domain to another
        top = self.convert_domain(top, src=src, dest=dest)
        btm = self.convert_domain(btm, src=src, dest=dest)
        # Compute the FWHM in the new domain
        new_fwhm = top - btm
        return new_fwhm
        
    def size(self, peak_fwhm: float, peak_position: float, domain="tth") -> float:
        """Calculate the particle size by Sherrer analysis.
        
        Parameters
        ==========
        peak_fwhm
          Full-width at half maximum (in degrees) of the peak.
        peak_position
          Position of the peak, in degrees if *domain* is "tth" or
          reciprocal angstroms if *domain* is "q".
          
        Returns
        =======
        size
            The effective size of coherent scattering domains,
            with units matching ``self.wavelength``.
        
        """
        # Make sure a usable domain was given
        if domain not in ["tth", "q"]:
            raise ValueError('*domain* must be either "tth" or "q" (got "{}")'
                             ''.format(domain))
        # Convert from q to 2θ
        peak_position_tth = self.convert_domain(peak_position, src=domain, dest="tth")
        peak_fwhm_tth = self.convert_fwhm_domain(peak_fwhm, peak_position=peak_position,
                                                 src=domain, dest="tth")
        # Correct for instrumental broadening
        instrument_fwhm = self.instrument_fwhm(twotheta=peak_position_tth)
        sample_fwhm = peak_fwhm_tth - instrument_fwhm
        # Now we work in radians to apply the Scherrer equation
        theta = np.radians(peak_position_tth / 2)
        beta = np.radians(sample_fwhm)
        size = self.shape_factor * self.wavelength / beta / np.cos(theta)
        return size
        
    def instrument_fwhm(self, *, twotheta: float=None, q: float=None) -> float:
        """Either *peak_twotheta* or *peak_q* is required.

        Assumes twotheta is in degrees.

        Returns
        =======
          Full width at half-maximum in degrees.

        """
        # Convert q to 2θ, if necessary
        if twotheta is None:
            twotheta = synthxrd.q_to_twotheta(q, wavelength=self.wavelength)
        theta = np.radians(twotheta / 2)
        s = np.sqrt(self.U * np.tan(theta)**2 + self.V * np.tan(theta) + self.W)
        g = self.Z + self.X / np.cos(theta) + self.Y * np.tan(theta)
        gamFW = lambda s,g: np.exp(np.log(s**5+2.69269*s**4*g+2.42843*s**3*g**2+4.47163*s**2*g**3+0.07842*s*g**4+g**5)/5.)
        return gamFW(2.35482*s,g) / 100.
