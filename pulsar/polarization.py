import numpy as np

def calculate_stokes_parameters(total_intensity, degree_of_polarization,
                                angle_degrees, threshold=1e-10):
    """
    Calculate the Stokes parameters Q and U based on the total intensity, degree of polarization,
    and polarization angle.
    
    Parameters:
    total_intensity (float): The total intensity (T or I).
    degree_of_polarization (float): The degree of linear polarization (P) as a percentage (0-100).
    angle_degrees (float): The polarization angle in degrees.
    threshold (float): Threshold below which values are considered zero.
    
    Returns:
    tuple: A tuple (T, Q, U) representing the Stokes parameters.
    """
    
    P = degree_of_polarization / 100.0
    polarized_intensity = P * total_intensity
    angle_radians = np.deg2rad(angle_degrees)
    
    # Calculate Q and U
    Q = polarized_intensity * np.cos(2 * angle_radians)
    U = polarized_intensity * np.sin(2 * angle_radians)
    
    # Zero out very low values
    Q = 0 if abs(Q) < threshold else Q
    U = 0 if abs(U) < threshold else U
    
    return (total_intensity, Q, U)
