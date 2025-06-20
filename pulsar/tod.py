from typing import Protocol, Tuple, Any

import numpy as np

class TOD(Protocol): # pragma: no cover
    """
    Protocol to represent Time-Ordered Data (TOD) for astronomical observations.

    This protocol ensures that any TOD implementation provides essential functionalities
    and data attributes necessary for astronomical data processing.
    """

    @property
    def id(self) -> str: # pragma: no cover
        """
        Unique identifier for the TOD instance.
        """
        pass

    @property
    def data(self) -> np.ndarray: # pragma: no cover
        """
        Retrieves the entire dataset as a NumPy array.

        Returns:
            np.ndarray: The complete time-ordered data.
        """
        pass

    @property
    def num_detectors(self) -> int: # pragma: no cover
        """
        Returns the number of detectors.
        """
        pass

    @property
    def num_samples(self) -> int: # pragma: no cover
        """
        Returns the number of samples in the dataset as an integer.
        """
        pass

    @property
    def sampling_rate(self) -> float: # pragma: no cover
        """
        Returns the sampling rate of the data in Hz.
        """
        pass
    
    def calibrate(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """
        Calibrates the TOD in-place according to some specified calibration procedure.
        Subclasses may define additional parameters.

        Args:
            *args: Positional arguments for subclass implementations.
            **kwargs: Keyword arguments for subclass implementations.
        """
        pass

    @property
    def calibrated(self, *args) -> bool: # pragma: no cover
        """Get the calibrated state (read-only)."""
        pass

    def downsample(self, factor: int) -> None: # pragma: no cover
        """
        Downsamples the TOD by a specified factor.
        """
        pass

    def remove_source(self, ra: float, dec: float, R: float) -> None: # pragma: no cover
        """
        Removes the influence of a source from the TOD.

        Parameters:
            ra (float): The right ascension of the source in radians.
            dec (float): The declination of the source in radians.
            R (float): The radius of the source in radians.
        """
        pass

    def locate_source(self, ra: float, dec: float, R: float) -> Tuple[np.ndarray, np.ndarray]: # pragma: no cover
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
        pass

    def locate_samples(self, ra: float, dec: float, R: float) -> np.ndarray: # pragma: no cover
        """
        Find the minimum and maximum values for each detector in the given region.

        Parameters:
        - ra (float): The right ascension of the source in radians.
        - dec (float): The declination of the source in radians.
        - R (float): The radius of the source in radians.

        Returns:
        - np.ndarray: An array containing the minimum and maximum values for each detector.
        """
        pass

    def multiplicative_gaussian_noise(self, sigma: float) -> None: # pragma: no cover
        """
        Adds multiplicative Gaussian noise to the TOD.

        Parameters:
            sigma (float): The standard deviation of the Gaussian noise.
        """
        pass

    def fill_gaps(self, data: np.ndarray = None): # pragma: no cover
        """
        Fills the gaps in the data using a specified gap-filling algorithm by the instrument.

        Parameters:
            data (np.ndarray): The data to fill the gaps in. If None, the original data is used.
        """
        pass