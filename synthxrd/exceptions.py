class NoBackgroundFile(ValueError):
    """A background file was not given to the constructor."""
    ...


class AmbiguousKphi(RuntimeError):
    """The KPHI for a scan is too close to several other kphi values."""
    ...


class UnknownHKL(KeyError):
    """The requested HKL value is not defined."""
    ...

class FileNotReadable(ValueError):
    """A file could not be read."""
    ...
