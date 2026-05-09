"""Stand-ins for `enlib`, `enact`, `mpi4py`, and `matplotlib` so that
`pulsar/act/*`, `pulsar/timing.py`, and `pulsar/search/*` can be imported and
exercised under test without the real ACT data libraries (which are not
installable on systems without ACT credentials).

Strategy:
- MagicMock at the package level (so unspecified attributes return MagicMocks).
- Real attributes overlaid on top for the surface area exercised by tests.
- Where pixell already provides a function (mjd2ctime, ang2rect, etc.) we
  delegate to it directly.

Public function: `build_mocks()` returns the dict that conftest patches into
`sys.modules`.
"""
from unittest.mock import MagicMock

import numpy as np
import pixell.fft as pixell_fft
import pixell.utils as pixell_utils


# -------------------------------- enlib --------------------------------

class _EnlibConfig:
    """A dict-backed stand-in for `enlib.config` (`set`/`get`)."""
    def __init__(self):
        self._store = {}

    def set(self, key, value):
        self._store[key] = value

    def get(self, key, default=None):
        return self._store.get(key, default)


class DataMissing(Exception):
    """Stand-in for `enlib.errors.DataMissing`."""


def _gapfill_joneig(data, samples, inplace=False):
    """Stand-in for `enlib.gapfill.gapfill_joneig`. No-ops the gap fill;
    returns the (possibly copied) data so callers see a real ndarray."""
    if inplace:
        return data
    return np.array(data, copy=True)


def _gapfill_linear(cut, data, inplace=False, transpose=False):
    """Stand-in for `enlib.sampcut.gapfill_linear`."""
    if inplace:
        return data
    return np.array(data, copy=True)


def _apply_window(tod, window, inverse=False):
    """Stand-in for `enlib.nmat.apply_window`. Mutates in place; returns nothing."""
    return None


def _coordinates_transform(*args, **kwargs):
    """Stand-in for `enlib.coordinates.transform`. Returns plausibly-shaped
    output (RA, Dec arrays) inferred from one of the positional inputs."""
    # `transform("hor", "cel", angles, time=..., site=...)` — return
    # arrays with the same length as `time`.
    time = kwargs.get('time')
    if time is None and len(args) >= 4:
        time = args[3]
    n = np.asarray(time).size if time is not None else 1
    return np.zeros(n), np.zeros(n)


def _interpol_pos(*args, **kwargs):
    """Stand-in for `enlib.coordinates.interpol_pos`. Returns a (ra, dec) pair."""
    return np.array([0.0, 0.0])


class _IersEntry:
    def __init__(self, mjd, dUT):
        self.mjd = mjd
        self.dUT = dUT


class _IersCvar:
    def __init__(self, n):
        self.iers_n = n


class _Iers:
    """Stand-in for `enlib.iers`. Provides three reference points so
    `_calc_leaps` can run without raising."""
    def __init__(self):
        self._entries = [
            _IersEntry(mjd=50000.0, dUT=12.0),
            _IersEntry(mjd=51000.0, dUT=13.0),
            _IersEntry(mjd=52000.0, dUT=14.0),
        ]
        self.cvar = _IersCvar(len(self._entries))

    def get(self, i):
        return self._entries[i]


# ----- pmat / sampcut ----------------------------------------------------

class _PmatPtsrcTransient:
    """Stand-in for `enlib.pmat.PmatPtsrcTransient`. Stores construction args
    and provides backward()/forward() that move zeros between data and params."""
    def __init__(self, profiles, scan, params, sys='cel'):
        self.profiles = profiles
        self.scan = scan
        self.params = params
        self.sys = sys

    def forward(self, data, params, pmul=1):
        # Mutate `data` in place so the calling convention matches.
        data[...] = 0
        return None

    def backward(self, data, params, pmul=1):
        # Leave params untouched (the caller supplies the slice it cares about).
        return None


class _PmatCut:
    def __init__(self, scan):
        self.scan = scan


class _PmatMap:
    def __init__(self, scan, omap):
        self.scan = scan
        self.omap = omap

    def backward(self, data, omap):
        return omap


# ----- nmat / nmat_measure ---------------------------------------------

class _NoiseMatrix:
    """Stand-in returned by `nmat_measure.NmatBuildDelayed.update`."""
    def __init__(self, ndet=2):
        self.ivar = np.full(ndet, 1.0)

    def apply_ft(self, ft, n, dtype):
        return None

    def white(self, tod):
        return None


class _NmatBuildDelayed:
    def __init__(self, model, cut=None, spikes=None):
        self.model = model
        self.cut = cut
        self.spikes = spikes

    def update(self, tod, srate):
        ndet = tod.shape[0] if hasattr(tod, 'shape') else 2
        return _NoiseMatrix(ndet=ndet)


def _detvecs_jon(ft, srate):
    """Stand-in for `nmat_measure.detvecs_jon`; returns an object with
    `apply(data)` that no-ops."""
    obj = MagicMock(name='Noise')
    obj.apply.side_effect = lambda data: None
    return obj


# ----- enmap (pixell has the real one but the original code uses enlib.enmap) -

class _Enmap:
    @staticmethod
    def geometry(coord, res, pixels):
        # Return (shape, wcs) — wcs is opaque to our code.
        return tuple(pixels), MagicMock(name='wcs')

    @staticmethod
    def zeros(shape, wcs, dtype):
        return np.zeros(shape, dtype=dtype)


# ----- enact ----------------------------------------------------------------

class _ActDataModule:
    """Stand-in for `enact.actdata`."""
    def __init__(self):
        self._calibrate_calls = []

    def calibrate(self, dataset, exclude=None):
        self._calibrate_calls.append((dataset, exclude))
        return None

    def read(self, entry, dets=None, verbose=False):
        # Returns a dataset stub with .ndet and .tod
        ds = MagicMock(name='Dataset')
        ds.ndet = len(dets) if dets is not None else 2
        ds.tod = np.zeros((ds.ndet, 16))
        return ds

    def offset_to_dazel(self, base, ref):
        return np.zeros((np.asarray(base).shape[0], 2))


def _avoidance_cut(boresight, point_offset, site, ra_dec, R):
    """Stand-in for `enact.cuts.avoidance_cut`. Returns an object with
    `to_mask()`, `to_list()`, and `ranges` so callers can get arrays back."""
    samples = MagicMock(name='AvoidanceCut')
    samples.to_mask.return_value = np.zeros(16, dtype=bool)
    samples.to_list.return_value = []
    samples.ranges = []
    return samples


# -------------------------- builder --------------------------

def build_mocks() -> dict:
    """Construct the dict of stand-in modules to install into sys.modules."""

    enlib = MagicMock(name='enlib')
    enlib_config = _EnlibConfig()
    # Mirror enlib.config's interface as both module-level functions and a sub-attribute.
    enlib.config = enlib_config

    enlib_utils = MagicMock(name='enlib.utils')
    # Delegate to pixell where possible.
    for name in (
        'mjd2ctime', 'ctime2mjd', 'mjd2jd', 'ang2rect', 'R_earth', 'c',
        'interp', 'nint', 'minmax', 'block_reduce', 'rewind', 'nowarn',
    ):
        setattr(enlib_utils, name, getattr(pixell_utils, name))

    enlib_fft = MagicMock(name='enlib.fft')
    enlib_fft.rfft = pixell_fft.rfft
    enlib_fft.irfft = pixell_fft.irfft
    enlib_fft.rfftfreq = pixell_fft.rfftfreq

    enlib_errors = MagicMock(name='enlib.errors')
    enlib_errors.DataMissing = DataMissing

    enlib_gapfill = MagicMock(name='enlib.gapfill')
    enlib_gapfill.gapfill_joneig = _gapfill_joneig

    enlib_sampcut = MagicMock(name='enlib.sampcut')
    enlib_sampcut.gapfill_linear = _gapfill_linear

    enlib_nmat = MagicMock(name='enlib.nmat')
    enlib_nmat.apply_window = _apply_window

    enlib_coordinates = MagicMock(name='enlib.coordinates')
    enlib_coordinates.default_site = MagicMock(name='default_site')
    enlib_coordinates.transform = _coordinates_transform
    enlib_coordinates.interpol_pos = _interpol_pos

    enlib_iers = _Iers()

    enlib_pmat = MagicMock(name='enlib.pmat')
    enlib_pmat.PmatPtsrcTransient = _PmatPtsrcTransient
    enlib_pmat.PmatCut = _PmatCut
    enlib_pmat.PmatMap = _PmatMap

    enlib_dataset = MagicMock(name='enlib.dataset')

    enlib_enmap = MagicMock(name='enlib.enmap')
    enlib_enmap.geometry = _Enmap.geometry
    enlib_enmap.zeros = _Enmap.zeros

    enact = MagicMock(name='enact')
    enact_actdata = _ActDataModule()

    enact_actscan = MagicMock(name='enact.actscan')
    # ACTScan factory — return a minimal scan-like object.
    def _act_scan(entry, subdets=None, verbose=False):
        scan = MagicMock(name='ACTScan')
        scan.dets = list(range(8 if subdets is None else len(subdets)))
        scan.ndet = len(scan.dets)
        scan.nsamp = 16
        scan.srate = 400.0
        scan.mjd0 = 50000.0
        scan.boresight = np.zeros((3, scan.nsamp))
        scan.point_offset = np.zeros((scan.ndet, 2))
        scan.tod = np.zeros((scan.ndet, scan.nsamp))
        scan.cut = MagicMock(name='cut')
        scan.cut_noiseest = MagicMock(name='cut_noiseest')
        scan.spikes = MagicMock(name='spikes')
        scan.site = MagicMock(name='site')
        scan.d = MagicMock(name='scan.d')
        scan.d.boresight = scan.boresight
        scan.d.point_offset = scan.point_offset
        scan.d.site = scan.site
        scan.d.point_correction = np.zeros(2)
        scan.d.point_template = np.zeros(2)
        return scan
    enact_actscan.ACTScan = _act_scan

    enact_cuts = MagicMock(name='enact.cuts')
    enact_cuts.avoidance_cut = _avoidance_cut

    class _FileDB:
        """Stand-in for `enact.filedb`."""
        def __init__(self):
            self.scans = MagicMock(name='filedb.scans')
            self.scans.select.side_effect = lambda q: MagicMock(ids=['id_1', 'id_2', 'id_3'])
            self.data = {}

        def init(self):
            return None

    enact_filedb = _FileDB()

    enact_nmat_measure = MagicMock(name='enact.nmat_measure')
    enact_nmat_measure.NmatBuildDelayed = _NmatBuildDelayed
    enact_nmat_measure.detvecs_jon = _detvecs_jon

    # `from enlib import X` resolves via getattr on the parent module first,
    # before falling back to sys.modules['enlib.X']. Wire up the submodule
    # attributes on the parent so attribute access returns the typed stubs.
    enlib.config = enlib_config
    enlib.coordinates = enlib_coordinates
    enlib.dataset = enlib_dataset
    enlib.enmap = enlib_enmap
    enlib.errors = enlib_errors
    enlib.fft = enlib_fft
    enlib.gapfill = enlib_gapfill
    enlib.iers = enlib_iers
    enlib.nmat = enlib_nmat
    enlib.pmat = enlib_pmat
    enlib.sampcut = enlib_sampcut
    enlib.utils = enlib_utils

    enact.actdata = enact_actdata
    enact.actscan = enact_actscan
    enact.cuts = enact_cuts
    enact.filedb = enact_filedb
    enact.nmat_measure = enact_nmat_measure

    return {
        'enact': enact,
        'enact.actdata': enact_actdata,
        'enact.actscan': enact_actscan,
        'enact.cuts': enact_cuts,
        'enact.filedb': enact_filedb,
        'enact.nmat_measure': enact_nmat_measure,
        'enlib': enlib,
        'enlib.config': enlib_config,
        'enlib.coordinates': enlib_coordinates,
        'enlib.dataset': enlib_dataset,
        'enlib.enmap': enlib_enmap,
        'enlib.errors': enlib_errors,
        'enlib.fft': enlib_fft,
        'enlib.gapfill': enlib_gapfill,
        'enlib.iers': enlib_iers,
        'enlib.nmat': enlib_nmat,
        'enlib.pmat': enlib_pmat,
        'enlib.sampcut': enlib_sampcut,
        'enlib.utils': enlib_utils,
        'mpi4py': MagicMock(name='mpi4py'),
        'mpi4py.MPI': MagicMock(name='mpi4py.MPI'),
    }
