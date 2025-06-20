from copy import copy
from dataclasses import dataclass, asdict
from typing import Tuple

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import numpy as np

from dataclasses import dataclass
import numpy as np

# TODO: rename Source to Target, and Target to Pulsar for clarity.
@dataclass
class Source:
    """
    A generic astronomical source with positional and size attributes.
    """
    name: str = ''          # Name of the source.
    alias: str = ''         # Alias of the source.
    ra: float = 0           # Right ascension in radians.
    dec: float = 0          # Declination in radians.
    amp: float = 0          # Amplitude (e.g., brightness) in milliJanskys.
    radius: float = 0       # Radius of the source in radians.

    def __post_init__(self):
        if self.amp < 0:
            raise ValueError("Amplitude must be positive.")
        if self.radius < 0:
            raise ValueError("Radius must be positive.")
        if self.ra < 0 or self.ra > 2 * np.pi:
            raise ValueError("Right ascension must be between 0 and 2*pi.")
        if self.dec < -np.pi / 2 or self.dec > np.pi / 2:
            raise ValueError("Declination must be between -pi/2 and pi/2.")

    def __str__(self):
        fields = [
            ("Name", self.name),
            ("Alias", self.alias),
            ("Right Ascension (ra)", f"{self.ra} radians"),
            ("Declination (dec)", f"{self.dec} radians"),
            ("Amplitude", f"{self.amp} mJy"),
            ("Radius", f"{self.radius} radians")
        ]
        lines = [f"{self.__class__.__name__}:"]
        lines += [f"  {label}: {value}" for label, value in fields if value not in (None, "")]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """
        Serializes the Source object to a dictionary.
        
        Returns:
            dict: Serialized dictionary of the Source object.
        """
        return asdict(self)

    def to_degrees(self) -> Tuple[float, float]:
        """
        Returns the right ascension and declination in degrees.

        Returns:
        - Tuple[float, float]: A tuple containing the right ascension and declination in degrees.
        """
        return np.rad2deg(self.ra), np.rad2deg(self.dec)

@dataclass
class Target(Source):
    """
    A specific type of source representing a pulsar with additional properties.
    """
    T: float = 0            # Rotational period in seconds.
    D: float = 0            # Duty cycle (pulse width divided by T).
    phi0: float = 0         # Initial phase.
    t0: float = 0           # Time offset for phase calculations.
    phase_sign: str = '-'   # Phase direction ('+' or '-').

    def __post_init__(self):
        super().__post_init__()  # Validate the Source attributes
        if self.D < 0 or self.D > 1:
            raise ValueError("Duty cycle must be between 0 and 1.")
        if self.phi0 < 0 or self.phi0 > 1:
            raise ValueError("Initial phase must be between 0 and 1.")
        if self.T <= 0:
            raise ValueError("Period must be positive.")
        if self.t0 < 0:
            raise ValueError("Time offset must be non-negative.")
        if self.phase_sign not in ['+', '-']:
            raise ValueError("Phase sign must be '+' or '-'.")

    def __str__(self):
        base_str = super().__str__()
        fields = [
            ("Period (T)", f"{self.T} s"),
            ("Duty Cycle (D)", f"{self.D * 100:.2f}%"),
            ("Initial Phase (phi0)", f"{self.phi0} radians"),
            ("Time Offset (t0)", f"{self.t0} s"),
            ("Phase Sign", self.phase_sign)
        ]
        extra_lines = [f"  {label}: {value}" for label, value in fields if value not in (None, "")]
        return base_str + "\n" + "\n".join(extra_lines)

    def offset_position(self, distance: float, direction: str = 'west',
                        unit: u.Unit = u.arcmin) -> 'Target':
        """
        Generate a new Target offset by a specified distance and direction.

        Parameters:
        distance (float): Distance to move.
        direction (str): Direction to move ('west', 'east', 'north', 'south'). Default is 'west'.
        unit (astropy.units.Unit): Unit of the distance. Default is arcminutes.

        Returns:
        Target: A new Target object with the offset position.
        """
        # Convert initial RA/Dec to SkyCoord object
        coord = SkyCoord(ra=self.ra * u.rad, dec=self.dec * u.rad, frame='icrs')
        
        # Define the distance as an Angle object
        separation = Angle(distance, unit=unit)
        
        # Determine the position angle based on the direction
        if direction == 'west':
            angle = 270
        elif direction == 'east':
            angle = 90
        elif direction == 'north':
            angle = 0
        elif direction == 'south':
            angle = 180
        else:
            raise ValueError("Invalid direction. Choose from 'west', 'east', 'north', or 'south'.")
        
        position_angle = Angle(angle, unit=u.deg)
        target = copy(self)

        # Calculate the new position
        new_coord = coord.directional_offset_by(position_angle, separation)

        # Extract the new RA and Dec in radians
        target.ra = new_coord.ra.radian
        target.dec = new_coord.dec.radian

        return target