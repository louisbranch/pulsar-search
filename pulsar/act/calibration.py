import pixell

# Beam calibration data updated on 2023-03-16 for DR6v4
# Description: Beam calibration for f090 (90 GHz) and other bands in the form of
# (calibration_value, uncertainty) in nanosteradians.
beam_calibrations = {
    "ar4": {"f150": (220.68, 1.95), "f220": (101.85, 1.49)},
    "ar5": {"f090": (467.52, 3.40), "f150": (211.47, 1.42)},
    "ar6": {"f090": (471.35, 2.57), "f150": (216.35, 1.05)}
}

def jansky_sr(freq: float = 98e9, cmb_T: float = 2.72548) -> float:
    """
    Calculate the calibration in Jansky/steradian for a given frequency. 
    This value is used to multiply by the TOD to convert it to a useful unit. 
    The default frequency is set to 98 GHz.

    Parameters:
    - freq (float): Frequency in Hz for which the calibration is calculated.
    - cmb_T (float): Temperature of the CMB.

    Returns:
    - float: Calibration value in Jansky/steradian.
    """
    return pixell.utils.dplanck(freq, cmb_T) / 1e3

def beam_sr(array: str, band: str = 'f090') -> float:
    """
    Calculate the beam calibration in steradians for a given array and band.
    This function retrieves calibration values to be multiplied by TOD for beam correction.

    Parameters:
    - array (str): Identifier of the array, such as 'ar4', 'ar5', or 'ar6'.
    - band (str): Frequency band for calibration. Defaults to 'f090'.

    Returns:
    - float: Beam calibration value in steradians.

    Raises:
    - ValueError: If beam calibration data is not found for the specified array and band.
    """
    if array not in beam_calibrations or band not in beam_calibrations[array]:
        raise ValueError(f"Beam calibration not found for {array} array and {band} band")

    nsr, _ = beam_calibrations[array][band]
    return nsr * 1e-9  # Convert from nanosteradians to steradians

def uK_to_mJy(amplitude_uK: float, array: str = 'ar6', band: str = 'f090',
              freq: float = 98e9, cmb_T: float = 2.72548) -> float:
    """
    Convert amplitude in microkelvins (uK) to milliJanskys (mJy).

    Parameters:
    - amplitude_uK (float): Amplitude in microkelvins (uK).
    - array (str): Array identifier, such as 'ar4', 'ar5', or 'ar6'. Defaults to 'ar6'.
    - band (str): Frequency band for calibration. Defaults to 'f090'.
    - freq (float): Frequency in Hz. Defaults to 98 GHz.
    - cmb_T (float): Temperature of the CMB. Defaults to 2.72548 K.

    Returns:
    - float: Amplitude in milliJanskys (mJy).
    """
    # Calculate Jansky per steradian
    jansky_per_sr = jansky_sr(freq, cmb_T)

    # Calculate beam size in steradians
    beam_in_sr = beam_sr(array, band)

    # Convert amplitude to mJy
    amplitude_mJy = amplitude_uK * jansky_per_sr * beam_in_sr

    return amplitude_mJy