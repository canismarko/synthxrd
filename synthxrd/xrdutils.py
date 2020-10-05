import numpy as np


def rotate_2d_image(data):
    data = np.swapaxes(data, 0, 1)
    data = data[::-1, ::-1]
    return data


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


def twotheta_to_q(twotheta, wavelength=None, energy=None):
    if wavelength is None and energy is None:
        raise ValueError("One of *wavelength* or *energy* is required.")
    # Convert from X-ray energy to X-ray wavelength
    if energy is not None:
        wavelength = energy_to_wavelength(energy)
    # Convert to q if necessary
    q = (4*np.pi/wavelength) * np.sin(np.radians(twotheta/2))
    return q


def q_to_twotheta(q, wavelength):
    radians = np.arcsin(q / 4 / np.pi * wavelength) * 2
    tth = np.degrees(radians)
    return tth
