import pytest
import numpy as np
from pulsar import Target, Source

class TestSource:
    def test_init(self):
        # wrong amplitude
        with pytest.raises(ValueError):
            Source(amp=-1.0)

        # wrong radius
        with pytest.raises(ValueError):
            Source(radius=-0.1)
        
        # wrong right ascension
        with pytest.raises(ValueError):
            Source(ra=-1.0)
            Source(ra=2*np.pi + 1.0)
        
        # wrong declination
        with pytest.raises(ValueError):
            Source(dec=-np.pi/2 - 1.0)
            Source(dec=np.pi/2 + 1.0)

    def test_str(self):
        source = Source(name='crab', amp=1.0, ra=1.0, dec=1.5, radius=0.1)
        expected_str = ("""Source:
  Name: crab
  Right Ascension (ra): 1.0 radians
  Declination (dec): 1.5 radians
  Amplitude: 1.0 mJy
  Radius: 0.1 radians""")
        assert str(source) == expected_str

    def test_as_dict(self):
        target = Target(
            T=1.0, D=0.5, phi0=0.0, amp=1.0, ra=1.0, dec=1.5, t0=0.0, phase_sign='-')
        expected_dict = {
            'name': '',
            'alias': '',
            'T': 1.0,
            'D': 0.5,
            'phi0': 0.0,
            'amp': 1.0,
            'ra': 1.0,
            'dec': 1.5,
            't0': 0.0,
            'phase_sign': '-',
            'radius': 0
        }
        assert target.as_dict() == expected_dict

class TestTarget:

    def test_init(self):
        # wrong duty cycle
        with pytest.raises(ValueError):
            Target(D=-0.1)

        # wrong amplitude
        with pytest.raises(ValueError):
            Target(amp=-1.0)

        # wrong radius
        with pytest.raises(ValueError):
            Target(radius=-0.1)
        
        # wrong phase sign
        with pytest.raises(ValueError):
            Target(phase_sign='a')

        # wrong initial phase
        with pytest.raises(ValueError):
            Target(phi0=-0.1)
            Target(phi0=1.1)

        # wrong period
        with pytest.raises(ValueError):
            Target(T=0.0)
            Target(T=-1.0)

        # wrong time offset
        with pytest.raises(ValueError):
            Target(t0=-1.0, T=1.0)

        # wrong right ascension
        with pytest.raises(ValueError):
            Target(ra=-1.0, T=1.0)
            Target(ra=2*np.pi + 1.0)
        
        # wrong declination
        with pytest.raises(ValueError):
            Target(dec=-np.pi/2 - 1.0, T=1.0)
            Target(dec=np.pi/2 + 1.0, T=1.0)

    def test_str(self):
        target = Target(name='crab', amp=1.0, ra=1.0, dec=1.5, radius=0.1, T=1.0, D=0.5, phi0=0.0, t0=0.0, phase_sign='-')
        expected_str = ("""Target:
  Name: crab
  Right Ascension (ra): 1.0 radians
  Declination (dec): 1.5 radians
  Amplitude: 1.0 mJy
  Radius: 0.1 radians
  Period (T): 1.0 s
  Duty Cycle (D): 50.00%
  Initial Phase (phi0): 0.0 radians
  Time Offset (t0): 0.0 s
  Phase Sign: -""")
        assert str(target) == expected_str

    def test_as_dict(self):
        target = Target(
            T=1.0, D=0.5, phi0=0.0, amp=1.0, ra=1.0, dec=1.5, t0=0.0, phase_sign='-')
        expected_dict = {
            'name': '',
            'alias': '',
            'T': 1.0,
            'D': 0.5,
            'phi0': 0.0,
            'amp': 1.0,
            'ra': 1.0,
            'dec': 1.5,
            't0': 0.0,
            'phase_sign': '-',
            'radius': 0
        }
        assert target.as_dict() == expected_dict

    def test_offset_position(self):
        target = Target(ra=0.0, dec=0.0, T=1)
        
        west = target.offset_position(1.0, direction='west')
        assert west.ra == pytest.approx(6.283, abs=1e-3)
        assert west.dec == pytest.approx(0.0, abs=1e-3)

        east = target.offset_position(1.0, direction='east')
        assert east.ra == pytest.approx(0.0, abs=1e-3)
        assert east.dec == pytest.approx(0.0, abs=1e-3)

        north = target.offset_position(1.0, direction='north')
        assert north.ra == pytest.approx(0.0, abs=1e-3)
        assert north.dec == pytest.approx(3e-4, abs=1e-3)

        south = target.offset_position(1.0, direction='south')
        assert south.ra == pytest.approx(0.0, abs=1e-3)
        assert south.dec == pytest.approx(-3e-4, abs=1e-3)

        # wrong direction
        with pytest.raises(ValueError):
            target.offset_position(1.0, direction='wrong')

    def test_to_degrees(self):
        target = Target(ra=np.pi/2, dec=np.pi/4, T=1.0)
        ra, dec = target.to_degrees()
        assert ra == 90.0
        assert dec == 45.0

        target = Target(ra=np.pi, dec=-np.pi/2, T=1.0)
        ra, dec = target.to_degrees()
        assert ra == 180.0
        assert dec == -90.0

        target = Target(ra=0, dec=0, T=1.0)
        ra, dec = target.to_degrees()
        assert ra == 0.0
        assert dec == 0.0
