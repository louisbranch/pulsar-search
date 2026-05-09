from typing import Tuple, List, Optional

import numpy as np
from enact import actscan, actdata, cuts
from enlib import dataset, gapfill, utils, coordinates

from .calibration import jansky_sr, beam_sr
from .. import log

class TOD:
    """
    A class to represent a Time-Ordered Data (TOD) object, which includes identification
    and associated scan, dataset, band, and array information for astronomical observations.

    Attributes:
        id (str): A unique identifier for the TOD.
        scan (enact.ACTScan): The scan metadata associated with this TOD.
        dataset (enlib.DataSet): The dataset containing the data for this TOD.
        band (str): The frequency band of the observations, e.g., 'f090'.
        array (str): The specific array configuration, e.g., 'ar5'.
    """

    def __init__(self, id: str, scan: actscan.ACTScan, dataset: dataset.DataSet, 
                 band: str, array: str):
        """
        Initializes the TOD object with the necessary information.

        Parameters:
            id (str): The unique identifier for this TOD.
            scan (enact.ACTScan): An object containing the metadata of the scan.
            dataset (enlib.DataSet): An object containing the actual data of the TOD.
            band (str): The frequency band of the observations.
            array (str): The specific array configuration used in the observations.
        """
        self._id = id
        self.scan = scan
        self.dataset = dataset
        self.band = band
        self.array = array
        self._calibrated = False

    @property
    def id(self) -> str:
        """
        Unique identifier for the TOD instance.
        """
        return self._id

    def __str__(self) -> str:
        """
        Returns the string representation of the TOD, which includes its identifier and band and array info.

        Returns:
            str: A string representation of this TOD object.
        """
        return f"TOD [{self.id}] - Band: {self.band}, Array: {self.array}"

    @property
    def num_detectors(self) -> int:
        """
        Returns the number of detectors.
        """
        return self.dataset.ndet

    @property
    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset as an integer.
        """
        return self.scan.nsamp

    @property
    def sampling_rate(self) -> float:
        """
        Returns the sampling rate of the data in Hz.
        """
        return self.scan.srate

    @property
    def data(self) -> np.ndarray:
        """
        Returns the calibrated data from the dataset.

        Returns:
            np.ndarray: The full time-ordered data.
        """
        if not self._calibrated:
            return self.dataset.tod
        return self.scan.tod

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """
        Sets the data in the dataset to the given data.

        Parameters:
            data (np.ndarray): The new data to set in the dataset.
        """
        self.scan.tod = data

    def calibrate(self, exclude_filters: Optional[List[str]] = None) -> None:
        """
        Calibrates the TOD in-place. First, it updates the dataset to micro kelvin
        using a function from the `actdata` module. Then, it calibrates the dataset to
        milli Jansky using calibration factors based on the array and band of the TOD.
        """
        if self._calibrated:
            return
        
        log.debug(f"Calibrating TOD: {self.id}")

        # Calibrate dataset to micro kelvin (updating in place)

        actdata.calibrate(self.dataset, exclude=exclude_filters)

        jansky_to_steradian = jansky_sr()
        beam_to_steradian = beam_sr(self.array, band=self.band)

        # Calibrate the dataset to milli Jansky by applying the calibration factors
        self.scan.tod = self.data * jansky_to_steradian * beam_to_steradian
        self._calibrated = True
    
    @property
    def calibrated(self) -> bool:
        """Get the calibrated state (read-only)."""
        return self._calibrated
    
    def downsample(self, factor: int) -> None:
        """
        Downsamples the TOD by a specified factor. This method modifies the scan data in-place,
        reducing its resolution by the given downsampling factor.

        Parameters:
            factor (int): The factor by which to downsample the TOD. Must be an integer greater than 1.
        """
        if factor > 1:
            self.scan = self.scan[:, ::factor]
        else:
            raise ValueError("Downsampling factor must be greater than 1.")

        self.data = self.scan.get_samples()

    def remove_source(self, ra: float, dec: float, R: float) -> None:
        """
        Removes the influence of a source from the TOD given the source coordinates and radius.

        Parameters:
            ra (float): The right ascension of the source in radians.
            dec (float): The declination of the source in radians.
            R (float): The radius of the source in radians.
        """

        scan = self.scan.d
        samples = cuts.avoidance_cut(scan.boresight, scan.point_offset, scan.site, (ra, dec), R)
        gapfill.gapfill_joneig(self.data, samples, inplace=True)

    def locate_source(self, ra: float, dec: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Locate a source from the TOD given the source coordinates and radius.

        Parameters:
            ra (float): The right ascension of the source in radians.
            dec (float): The declination of the source in radians.
            R (float): The radius of the source in radians.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - The cleaned time-ordered data with the source's influence removed.
                - A boolean array indicating which samples are within the source avoidance region.
        """
        scan = self.scan.d
        samples = cuts.avoidance_cut(scan.boresight, scan.point_offset, scan.site, (ra, dec), R)
        model = gapfill.gapfill_joneig(self.data, samples, inplace=False)

        return model, samples

    def locate_samples(self, ra: float, dec: float, R: float) -> np.ndarray:
        """
        Find the first and last sample indices for each detector that fall within
        the given sky region.

        Parameters:
        - ra (float): The right ascension of the source in radians.
        - dec (float): The declination of the source in radians.
        - R (float): The radius of the source in radians.

        Returns:
        - np.ndarray: An (n_detectors, 2) array of [min_index, max_index] pairs.
        """

        min_max_pairs = []
        _, cuts = self.locate_source(ra, dec, R)
        for cut in cuts.to_list():
            if len(cut) == 0:
                continue
            min_values = cut[:, 0]
            max_values = cut[:, 1]
            lowest_min = np.min(min_values)
            highest_max = np.max(max_values)
            min_max_pairs.append([lowest_min, highest_max])
        return np.array(min_max_pairs, dtype=int)

    def recover_position(self, sample_idx: int, det_idx: int = 0) -> np.ndarray:
        """
        Recovers the RA/Dec of a given sample for a specific detector.

        Parameters:
            sample_idx (int): The index of the sample to recover the position for.
            det_idx (int): The detector index. Defaults to 0 (the first detector).

        Returns:
            np.ndarray: An array [ra, dec] (in radians) of the recovered position.
        """
        return self.recover_radec(
            sample_idx, det_idx, self.scan.boresight, self.scan.point_offset, self.scan.site
        )

    @staticmethod
    def recover_radec(sample_idx, det_idx, bore, det_offs, site):
        """Recover the RA/Dec for a given sample index and detector index."""
        mjd = utils.ctime2mjd(bore[0, sample_idx])  # Get MJD at that sample time

        # Interpolate bore-sight pointing at this timestamp
        bore_radec = coordinates.interpol_pos("tele", "cel", bore[1:, sample_idx], mjd, site)

        # Extract detector offset for the given detector index
        det_x, det_y = det_offs[det_idx]  # Extract (x, y) offset

        # Apply correction for declination to avoid RA stretching
        cosel = np.cos(bore_radec[1])  # Cosine of declination

        # Convert telescope offsets to celestial coordinates
        ra_offset = det_x / cosel
        dec_offset = det_y

        # Compute final detector RA/Dec
        recovered_ra = bore_radec[0] + ra_offset
        recovered_dec = bore_radec[1] + dec_offset

        # Normalize RA to wrap correctly
        recovered_ra = utils.rewind(recovered_ra, bore_radec[0])

        return np.array([recovered_ra, recovered_dec])

    def multiplicative_gaussian_noise(self, sigma: float) -> None:
        """
        Adds multiplicative Gaussian noise to the TOD.

        Parameters:
            sigma (float): The standard deviation of the Gaussian noise.
        """
        size = self.data.shape[0]
        self.data *= np.random.normal(1, sigma, size)[:, None]

    def fill_gaps(self, data: np.ndarray = None):
        """
        Fills the gaps in the data using the Jon's eigenmode iteration.

        Parameters:
            data (np.ndarray): The data to fill the gaps in. If None, the TOD data is used.
        """
        data = self.data if data is None else data
        gapfill.gapfill_joneig(data, self.scan.cut, inplace=True)