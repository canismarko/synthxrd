class NoBackgroundFile(ValueError):
    """A background file was not given to the constructor."""
    ...


class AmbiguousKphi(RuntimeError):
    """The KPHI for a scan is too close to several other kphi values."""
    ...
