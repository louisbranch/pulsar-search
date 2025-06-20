from abc import ABC, abstractmethod
import tempfile
from urllib.request import urlretrieve
from typing import Optional

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from pixell import utils
from enlib import coordinates, iers # FIXME: move enlib dependency to a separate module for ACT-specific code

from .target import Target

class TimingModel(ABC):
    """
    Abstract base class for timing models used in pulsar signal analysis.
    """

    @abstractmethod
    def obstime2phase(self, ctime: float) -> float:
        """
        Convert observation time to pulsar phase.
        """
        pass

    def __str__(self):
        return self.__class__.__name__


class PeriodTimingModel(TimingModel):
    """
    A simple timing model assuming constant pulsar period and phase offset.

    This model does not account for light travel time, relativistic corrections,
    or barycentric time conversions. It is useful for synthetic or test profiles
    where precise timing is not required.
    """
    def __init__(self, target):
        self.target = target

    def obstime2phase(self, ctime):
        """
        Convert observation time to pulsar phase using a constant-period model.
        Phase is computed as a linear function of time, modulo 1.
        """
        t = self.target
        if t.phase_sign == "-":
            return (t.phi0 - (ctime - t.t0) / t.T) % 1
        else:
            return (t.phi0 + (ctime - t.t0) / t.T) % 1


class BarycentricTimingModel(TimingModel):
    """
    A timing model that applies barycentric corrections to observation time
    before computing pulsar phase.

    Includes:
    - UTC → TAI (adds leap seconds)
    - TAI → TDT (adds constant offset)
    - TDT → TDB (adds relativistic correction from Earth's orbit)
    - Geometric delay from observer to solar system barycenter
    - Frequency evolution based on timing solution
    """
    def __init__(self, target: Target, solution_file: str, site = coordinates.default_site,
                 ephem: Optional[str] = None, delay=0):
        self.target = target
        self.site = site
        self.ephem = self._resolve_ephem(ephem)
        self.delay = delay
        self._load_timing_data(solution_file)

    def obstime2phase(self, ctime):
        """
        Compute pulsar phase from observation time with full timing corrections.
        """
        obs_delay = self._calc_obs_delay(ctime, self.target, self.site, ephem=self.ephem)
        tdb = self._tdt2tdb(self._tai2tdt(self._utc2tai(ctime)))
        tdb -= obs_delay
        tdb -= self.delay
        ind, x = self._calc_ind_x(tdb)
        return x % 1

    def _resolve_ephem(self, ephem: Optional[str]) -> Optional[str]:
        """
        Resolve ephemeris: if a URL is provided, download it locally and return the path.
        Otherwise return the ephem string unchanged.
        """
        if ephem and ephem.startswith("http"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bsp")
            urlretrieve(ephem, tmp.name)
            return tmp.name
        return ephem

    def _load_timing_data(self, solution_file):
        """
        Load a pulsar timing solution file with columns for reference times,
        time offsets, spin frequency and its derivative.

        Expected file format (space-delimited ASCII text):
        # Date     MJD   t_JPL  t_acc   nu         sigma_nu   nudot     sigma_nudot DM    tau_408
        #            |      |     |      |             |         |          |         |       |
        #            |      |     |      |             |         |          |         |       └─ (ignored)
        #            |      |     |      |             |         |          |         └─ DM (ignored)
        #            |      |     |      |             |         |          └─ Frequency derivative (nudot)
        #            |      |     |      |             |         └─ Frequency (nu)
        #            |      |     |      └─ Time offset (t0, in seconds)
        #            └─ MJD reference time (tref)

        The parser uses only these columns:
        - Column 3: Time offset t0 (sec)
        - Column 4: Ignored (t_acc)
        - Column 6: Spin frequency (Hz)
        - Column 8: Spin-down rate (nudot), assumed in units of 1e-15 s⁻²
        """
        data = np.loadtxt(solution_file, usecols=(3, 4, 6, 8), ndmin=2).T
        self.tref = utils.mjd2ctime(data[0])
        self.t0 = data[1]
        self.freq = data[2]
        self.dfreq = data[3] * 1e-15
        self.P = 1 / self.freq
        self.dP = -1 / self.freq ** 2 * self.dfreq

    def _calc_ind_x(self, ctime):
        """
        Locate the appropriate index in timing reference table,
        and calculate the corresponding pulsar phase evolution variable x.
        """
        ind = np.searchsorted(self.tref, ctime, side="right") - 1
        x = self._calc_x(ctime - (self.tref[ind] + self.t0[ind]), self.P[ind], self.dP[ind])
        return ind, x

    def _calc_x(self, t, P, dP):
        """Compute phase evolution variable x from time t, period P, and period derivative dP."""
        return np.log1p(dP / P * t) / dP

    def _calc_obs_delay(self, ctime, target, site, ephem=None, interp=True, step=10):
        """
        Compute geometric light travel delay from observer to barycenter.
        """
        if interp:
            t1, t2 = utils.minmax(ctime)
            tsamp = np.linspace(t1, t2, np.ceil((t2 - t1) / step).astype(int))
            dsamp = self._calc_obs_delay(tsamp, target, site, ephem=ephem, interp=False)
            return utils.interp(ctime, tsamp, dsamp)

        ctime = np.asarray(ctime).reshape(-1)

        if ephem is not None:
            solar_system_ephemeris.set(ephem)

        ra_dec = np.array([target.ra, target.dec])
        vec_obsdir = utils.ang2rect(ra_dec)
        obs_ra, obs_dec = coordinates.transform("hor", "cel", [0, np.pi/2], time=utils.ctime2mjd(ctime), site=site)
        vec_earth_obs = utils.ang2rect([obs_ra, obs_dec]) * utils.R_earth
        vec_bary_earth = get_body_barycentric("earth", Time(ctime, format="unix")).xyz.to("m").value
        vec_bary_obs = vec_bary_earth + vec_earth_obs
        vec_obs_bary = -vec_bary_obs
        delay = np.sum(vec_obs_bary * vec_obsdir[:, None], 0) / utils.c
        return delay.reshape(ctime.shape)

    def _utc2tai(self, ctime):
        """Convert UTC to TAI using leap seconds."""
        return ctime + self._calc_leaps(ctime)

    def _calc_leaps(self, ctime):
        """Compute number of leap seconds at given time using IERS data."""
        if not hasattr(self, '_utctai'):
            iers_mjds = np.array([iers.get(i).mjd for i in range(iers.cvar.iers_n)])
            dUTs = np.array([iers.get(i).dUT for i in range(iers.cvar.iers_n)])
            leap_inds = np.where(utils.nint(dUTs[1:] - dUTs[:-1]) != 0)[0] + 1
            self._leap_mjds = iers_mjds[leap_inds]
            self._leap_vals = 12 + np.cumsum(utils.nint(dUTs[1:] - dUTs[:-1]))[leap_inds - 1]
        ind = np.maximum(np.searchsorted(self._leap_mjds, utils.ctime2mjd(ctime), side="right") - 1, 0)
        return self._leap_vals[ind]

    def _tai2tdt(self, ctime):
        """
        Convert TAI to TDT (Terrestrial Dynamical Time).
        This uses the IAU-defined offset of 32.184 seconds, representing the historical
        difference between Ephemeris Time (ET) and TAI.
        """
        return ctime + 32.184

    def _tdt2tdb(self, ctime):
        """
        Convert TDT to TDB (Barycentric Dynamical Time).
        Adds periodic relativistic correction from Earth's orbit.
        """
        return ctime + self._calc_tdb_off(ctime)

    def _ctime2jd(self, ctime):
        """Convert UNIX time to Julian Date."""
        return utils.mjd2jd(utils.ctime2mjd(ctime))

    def _calc_tdb_off(self, ctime):
        """
        Compute the TDB - TDT offset using a sinusoidal approximation
        of relativistic time dilation from Earth's elliptical orbit.
        """
        jd = self._ctime2jd(ctime)
        T = (jd - 2451545.0) / 365.25
        g = 2 * np.pi * (357.528 + 359.99050 * T) / 360
        return 0.001658 * np.sin(g + 0.0167 * np.sin(g))