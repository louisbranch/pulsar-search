import numpy as np
from typing import List, Optional, Tuple

from enlib import pmat, sampcut, utils
from enact import actdata

from .tod import TOD

class PmatTotTransient:
    def __init__(self, tod: TOD, srcs: List[object], ndir: int = 1, perdet: bool = False, sys: str = "cel",
                 ncomp: int = 3, beam_size: Tuple[float, float] = (1, 1)):
        """
        Initializes the pointing matrix for transient sources, handling the calculation
        of pointing vectors for pulsar observations within time-domain detector data.

        Parameters:
        - tod (TOD): Time-ordered data object containing the detector data and scan information.
        - srcs (List[object]): List of source objects, each providing celestial position and light curve profile.
        - ndir (int): Number of pointing directions to calculate per source.
        - perdet (bool): If True, separate parameters are generated for each detector.
        - sys (str): Coordinate system used for the pointing calculations, default is celestial ("cel").
        - ncomp (int): Number of components in the amplitude parameters, default is 3: [T, Q, U].
        - beam_size (Tuple[float, float]): Beam size, default is (1, 1) for circular beam.
        """

        self.ncomp = ncomp

        scan = tod.scan
        self.params = np.zeros([len(srcs), ndir, scan.ndet if perdet else 1, 8], dtype=float)
        srcpos = np.array([p.src[:2] for p in srcs]).T  # Extract RA, DEC
        self.params[:, :, :, :2] = srcpos[::-1, None, None, :].T  # Swap RA and DEC, align dimensions
        self.params[:, :, :, 5:7] = beam_size  # Assume a circular beam

        ctime = utils.mjd2ctime(scan.mjd0) + scan.boresight[:, 0]
        profiles = np.array([p.obstime2profile(ctime) for p in srcs])
        self.psrc = pmat.PmatPtsrcTransient(profiles, scan, self.params, sys=sys)
        self.pcut = pmat.PmatCut(scan)
        
        # Extract basic offset. Warning: referring to scan.d is fragile, since
        # scan.d is not updated when scan is sliced
        self.off0 = scan.d.point_correction
        self.off = np.zeros_like(self.off0)
        self.el = np.mean(scan.boresight[::100, 2])
        self.point_template = scan.d.point_template
        self.cut = scan.cut

    def set_offset(self, off: float):
        """
        Sets and applies a new pointing offset to the existing scan offsets.

        Parameters:
        - off (float): The offset to be added to the existing pointing corrections.
        """
        self.off = off * 1
        self.psrc.scan.offsets[:, 1:] = actdata.offset_to_dazel(self.point_template + off + self.off0, [0, self.el])

    def forward(self, data: np.ndarray, amps: np.ndarray, pmul: float = 1):
        """
        Applies the pointing matrix operations to project the source signals into the time-domain data.

        Parameters:
        - data (np.ndarray): Time-domain data array to be modified in-place.
        - amps (np.ndarray): Amplitude parameters for each source and direction.
            amps should be [nsrc,ndir,ndet|1, ncomp], where ncomp is [T,Q,U] or [T].
        - pmul (float): Multiplicative factor for the amplitude parameters.
        """
        params = self.params.copy()
        params[..., 2:2 + amps.shape[-1]] = amps
        self.psrc.forward(data, params, pmul=pmul)
        sampcut.gapfill_linear(self.cut, data, inplace=True)

    def backward(self, data: np.ndarray, amps: Optional[np.ndarray] = None, pmul: float = 1) -> np.ndarray:
        """
        Extracts amplitude parameters from the time-domain data by inverting the pointing matrix operations.

        Parameters:
        - data (np.ndarray): Time-domain data from which to extract amplitudes.
        - amps (Optional[np.ndarray], optional): Array to store extracted amplitude parameters. If None, a new array is created.
        - pmul (float): Multiplicative factor for the amplitude parameters.

        Returns:
        - np.ndarray: Extracted or updated amplitude parameters.
        """
        params = self.params.copy()
        data = sampcut.gapfill_linear(self.cut, data, inplace=False, transpose=True)
        self.psrc.backward(data, params, pmul=pmul)
        if amps is None:
            amps = params[..., 2:2 + self.ncomp]
        else:
            amps[:] = params[..., 2:2 + amps.shape[-1]]
        return amps