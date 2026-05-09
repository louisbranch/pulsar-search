import pytest
import pixell

from pulsar.act.calibration import beam_sr, jansky_sr, uK_to_mJy, beam_calibrations


class TestJanskySr:
    def test_delegates_to_pixell_dplanck(self):
        # jansky_sr returns dplanck/1e3
        expected = pixell.utils.dplanck(98e9, 2.72548) / 1e3
        assert jansky_sr() == pytest.approx(expected)

    def test_accepts_custom_freq_and_temperature(self):
        out = jansky_sr(freq=150e9, cmb_T=2.7)
        expected = pixell.utils.dplanck(150e9, 2.7) / 1e3
        assert out == pytest.approx(expected)


class TestBeamSr:
    def test_known_array_and_band(self):
        nsr, _ = beam_calibrations['ar5']['f090']
        assert beam_sr('ar5', 'f090') == pytest.approx(nsr * 1e-9)

    def test_unknown_array_raises(self):
        with pytest.raises(ValueError, match='Beam calibration not found'):
            beam_sr('ar99', 'f090')

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match='Beam calibration not found'):
            beam_sr('ar5', 'f042')

    def test_default_band_is_f090(self):
        assert beam_sr('ar5') == beam_sr('ar5', 'f090')


class TestUkToMjy:
    def test_composes_jansky_and_beam(self):
        result = uK_to_mJy(1.0, array='ar5', band='f090')
        expected = jansky_sr(98e9, 2.72548) * beam_sr('ar5', 'f090')
        assert result == pytest.approx(expected)

    def test_scales_linearly_with_amplitude(self):
        a = uK_to_mJy(1.0, array='ar6', band='f150')
        b = uK_to_mJy(5.0, array='ar6', band='f150')
        assert b == pytest.approx(5.0 * a)
