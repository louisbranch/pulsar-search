from enlib import nmat, config, fft, utils
from enact import nmat_measure

class NmatTot:
    """
    A class to model and apply noise characteristics to Time-Ordered Data (TOD) based on 
    provided models and configurations.

    Attributes:
        nmat (NmatBuildDelayed): Noise matrix constructed and updated from the input scan data.
        model (str): The noise model configuration.
        window (float): The windowing value applied to the data, scaled by sample rate.
        ivar (np.ndarray): Inverse variance derived from the noise matrix.
        cut (array-like): Data cut configuration from the scan.
        filter (np.ndarray, optional): Frequency filter applied during the `apply` method, if specified.
    """

    def __init__(self, scan, model=None, window=None, filter=None):
        """
        Initializes the NmatTot class with scan data and optional configurations.

        Parameters:
            scan (ScanObject): The scan object containing the TOD and metadata.
            model (str, optional): The specific noise model to use, defaults to a configuration setting (jon).
            window (float, optional): The windowing factor, defaults to a configuration setting and scaled by sample rate.
            filter (tuple, optional): A tuple of (fknee, alpha) to create a frequency-dependent filter.
        """
        self.model = config.get("noise_model", model)
        self.window = config.get("tod_window", window) * scan.srate
        nmat.apply_window(scan.tod, self.window)

        self.nmat = nmat_measure.NmatBuildDelayed(self.model, cut=scan.cut_noiseest, spikes=scan.spikes)
        self.nmat = self.nmat.update(scan.tod, scan.srate)

        nmat.apply_window(scan.tod, self.window, inverse=True)
        self.ivar = self.nmat.ivar
        self.cut = scan.cut

        if filter:
            freq = fft.rfftfreq(scan.nsamp, 1 / scan.srate)
            fknee, alpha = filter
            with utils.nowarn():
                self.filter = (1 + (freq / fknee) ** -alpha) ** -1
        else:
            self.filter = None

    def apply(self, tod):
        """
        Applies the noise model and filter (if specified) to the given Time-Ordered Data.

        Parameters:
            tod (np.ndarray): The array of time-ordered data to which the noise model is applied.

        Returns:
            np.ndarray: The modified time-ordered data after applying the noise model and filter.
        """
        nmat.apply_window(tod, self.window)
        ft = fft.rfft(tod)
        self.nmat.apply_ft(ft, tod.shape[-1], tod.dtype)
        if self.filter is not None:
            ft *= self.filter
        fft.irfft(ft, tod, flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'])
        nmat.apply_window(tod, self.window)
        return tod

    def white(self, tod):
        """
        Applies a white noise model to the Time-Ordered Data using the window setting.

        Parameters:
            tod (np.ndarray): The array of time-ordered data to which the white noise model is applied.
        """
        nmat.apply_window(tod, self.window)
        self.nmat.white(tod)
        nmat.apply_window(tod, self.window)
