from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pulsar.flux_estimator import FluxEstimator
from .test_mocks import create_mock_tod

@pytest.fixture
def fe():
    return FluxEstimator(None)

class TestFluxEstimator:

    def test_match_fluxes(self):
        flux_estimator = FluxEstimator(None)

        mean_fluxes = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        errors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        # Single component [T]
        target = (4.2,)
        result = flux_estimator.match_fluxes(mean_fluxes, errors, target)

        assert result['profile'] == 1
        assert result['T'] == 4.0
        assert result['err'] == 0.4
        assert result['difference'] == 4.0 - 4.2
        assert result['chi_square'] == ((4.0 - 4.2)/ 0.4) ** 2
        assert result['reduced_chi_square'] == ((4.0 - 4.2)/ 0.4) ** 2 / 2

        # Multiple components [T, Q, U]
        target = np.array([7.5, 6.0, 7.0])
        result = flux_estimator.match_fluxes(mean_fluxes, errors, target)

        assert result['profile'] == 2
        assert result['T'] == 7.0
        assert result['err'] == 0.7
        assert result['difference'] == 7.0 - 7.5
        assert result['chi_square'] == ((7.0 - 7.5)/ 0.7) ** 2
        assert result['reduced_chi_square'] == ((7.0 - 7.5)/ 0.7) ** 2 / 2

    def test_extract_tod_data(self):
        flux_estimator = FluxEstimator(None)

        tod_id = '1234'
        dataframes = [
            pd.DataFrame({
                'tod_id': ['1234', '5678'],
                'rhs': [np.array([1, 2, 3]), np.array([4, 5, 6])],
                'div': [np.array([7, 8, 9]), np.array([10, 11, 12])]
            }),
            pd.DataFrame({
                'tod_id': ['1234', '5678'],
                'rhs': [np.array([13, 14, 15]), np.array([16, 17, 18])],
                'div': [np.array([19, 20, 21]), np.array([22, 23, 24])]
            })
        ]

        result = flux_estimator.extract_tod_data(tod_id, dataframes)

        assert not result.empty
        assert len(result) == 2
        assert all(result['tod_id'] == '1234')
        assert result.iloc[0]['source'] == 'target'
        assert result.iloc[1]['source'] == 'null_1'
        assert (result.iloc[0]['rhs'] == np.array([1, 2, 3])).all()
        assert (result.iloc[1]['rhs'] == np.array([13, 14, 15])).all()

    def test_extract_tod_data_empty(self):
        flux_estimator = FluxEstimator(None)

        tod_id = '9999'
        dataframes = [
            pd.DataFrame({
                'tod_id': ['1234', '5678'],
                'rhs': [np.array([1, 2, 3]), np.array([4, 5, 6])],
                'div': [np.array([7, 8, 9]), np.array([10, 11, 12])]
            }),
            pd.DataFrame({
                'tod_id': ['1234', '5678'],
                'rhs': [np.array([13, 14, 15]), np.array([16, 17, 18])],
                'div': [np.array([19, 20, 21]), np.array([22, 23, 24])]
            })
        ]

        result = flux_estimator.extract_tod_data(tod_id, dataframes)

        assert result.empty
        assert list(result.columns) == ['tod_id', 'rhs', 'div', 'source']

    def test_calculate_flux_3components(self, fe):
        # One profile (n_profiles=1) with 3 components (n_components=3).
        # Use diagonal 'div' for simple inversion:
        # For each component, flux = rhs / diag(div)
        #   rhs = [[6, 9, 12]]
        #   div = [[[2, 0, 0],
        #           [0, 3, 0],
        #           [0, 0, 4]]]
        # Expected flux = [[6/2, 9/3, 12/4]] = [[3, 3, 3]]
        dtype = [('tod_id', int), ('rhs', float, (1, 3)), ('div', float, (1, 3, 3))]
        data = np.array([
            (0, np.array([[6., 9., 12.]]), np.array([[[2., 0., 0.],
                                                    [0., 3., 0.],
                                                    [0., 0., 4.]]]))
        ], dtype=dtype)

        raw_fluxes, fluxes, mean_flux, err, snr = fe.calculate_flux(data, nsplits=1, subtract_median_flux=False)

        np.testing.assert_allclose(raw_fluxes[0], np.array([[3., 3., 3.]]))
        np.testing.assert_allclose(fluxes[0], np.array([[3., 3., 3.]]))
        np.testing.assert_allclose(mean_flux, np.array([[3., 3., 3.]]))
        np.testing.assert_allclose(err, np.array([[0.707107, 0.57735, 0.5]]), rtol=1e-5)
        np.testing.assert_allclose(snr, np.array([[4.24264, 5.19615, 6.]]), rtol=1e-5)

    def test_calculate_flux_3components_median(self, fe):
        # Two profiles (n_profiles=2) with 3 components (n_components=3)
        # We set the data so that each TOD produces a different flux for each profile.
        # For TOD 0:
        #   For profile 0: flux = rhs[0] / 1 = [2, 4, 6]
        #   For profile 1: flux = rhs[1] / 1 = [3, 6, 9]
        # For TOD 1:
        #   For profile 0: flux = [4, 8, 12]
        #   For profile 1: flux = [6, 12, 18]
        #
        # With nsplits = 2, each TOD is in its own split.
        # Mean flux is computed per profile:
        #   profile 0: ([2+4] / 2, [4+8] / 2, [6+12] / 2) = [3, 6, 9]
        #   profile 1: ([3+6] / 2, [6+12] / 2, [9+18] / 2) = [4.5, 9, 13.5]
        #
        # The median (computed along profiles for each component) is:
        #   comp0: median = (3+4.5)/2 = 3.75, comp1: 7.5, comp2: 11.25.
        # So, final mean_flux becomes:
        #   profile 0: [3 - 3.75, 6 - 7.5, 9 - 11.25] = [-0.75, -1.5, -2.25]
        #   profile 1: [4.5 - 3.75, 9 - 7.5, 13.5 - 11.25] = [0.75, 1.5, 2.25]

        dtype = [('tod_id', int), ('rhs', float, (2, 3)), ('div', float, (2, 3, 3))]
        
        # TOD 0: flux = rhs since div is identity.
        rhs0 = np.array([[2., 4., 6.],
                        [3., 6., 9.]])
        div0 = np.array([[[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]],
                        [[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]]])
        
        # TOD 1: flux = rhs since div is identity.
        rhs1 = np.array([[4., 8., 12.],
                        [6., 12., 18.]])
        div1 = np.array([[[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]],
                        [[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]]])
        
        data = np.array([(0, rhs0, div0), (1, rhs1, div1)], dtype=dtype)
        
        raw_fluxes, fluxes, mean_flux, err, snr = fe.calculate_flux(
            data, nsplits=2, subtract_median_flux=True
        )
        
        expected_mean_flux = np.array([[-0.75, -1.5, -2.25], [ 0.75,  1.5,  2.25]])
        np.testing.assert_allclose(mean_flux, expected_mean_flux, rtol=1e-5)


    def test_calculate_flux_invalid_splits(self, fe):
        dtype = [('tod_id', int), ('rhs', float, (1, 3)), ('div', float, (1, 3, 3))]
        data = np.array([
            (0, np.array([[6., 9., 12.]]), np.array([[[2., 0., 0.],
                                                    [0., 3., 0.],
                                                    [0., 0., 4.]]]))
        ], dtype=dtype)
        
        with pytest.raises(ValueError):
            fe.calculate_flux(data, nsplits=2)

    def test_calculate_flux_bad_tod(self, fe):

        dtype = [('tod_id', int), ('rhs', float, (1, 3)), ('div', float, (1, 3, 3))]
        data = np.array([
            (0, np.array([[6., 9., 12.]]), np.array([[[2., 0., 0.],
                                                    [0., 3., 0.],
                                                    [0., 0., 4.]]])),
            (1, np.array([[1., 1., 1.]]), np.array([[[np.inf, 0., 0.],
                                                    [0., 0., 0.],
                                                    [0., 0., 0.]]])),
        ], dtype=dtype)


        raw_fluxes, fluxes, mean_flux, err, snr = fe.calculate_flux(data, nsplits=2)

        np.testing.assert_allclose(raw_fluxes[1], np.array([[0, 0, 0]]))

    def test_calculate_flux_single_split_with_zero_diagonal(self, fe):
        # nsplits=1 path with subtract_median; zero on the diagonal exercises
        # the np.where(diagonal_div == 0, 1e-10, ...) protection.
        dtype = [('tod_id', int), ('rhs', float, (1, 3)), ('div', float, (1, 3, 3))]
        data = np.array([
            (0, np.array([[1., 2., 3.]]), np.array([[[1., 0., 0.],
                                                    [0., 0., 0.],
                                                    [0., 0., 4.]]]))
        ], dtype=dtype)

        raw_fluxes, fluxes, mean_flux, err, snr = fe.calculate_flux(
            data, nsplits=1, subtract_median_flux=True
        )

        # Error for the zero-diagonal component should be the 1/sqrt(1e-10) fallback.
        assert err[0, 1] == pytest.approx(1.0 / np.sqrt(1e-10))
        assert np.isfinite(err).all()

    def test_calculate_flux_zero_error_zeroes_snr(self, fe):
        # nsplits=1, finite diagonal, but mean_flux ends up at zero after median
        # subtraction. Force the 'np.all(err == 0)' SNR fallback by making the
        # diagonal large enough that 1/sqrt(diag) underflows... easier path:
        # patch err at the boundary by checking nsplits behavior with identical
        # split fluxes, so std is 0 across splits.
        dtype = [('tod_id', int), ('rhs', float, (1, 1)), ('div', float, (1, 1, 1))]
        data = np.array([
            (0, np.array([[2.]]), np.array([[[1.]]])),
            (1, np.array([[2.]]), np.array([[[1.]]])),
        ], dtype=dtype)
        raw_fluxes, fluxes, mean_flux, err, snr = fe.calculate_flux(
            data, nsplits=2, subtract_median_flux=False
        )
        # Identical fluxes across splits -> std is zero -> err is zero -> snr stays zero.
        np.testing.assert_array_equal(err, np.zeros_like(err))
        np.testing.assert_array_equal(snr, np.zeros_like(snr))

    def test_match_fluxes_zero_errors(self):
        # Zero errors: argmin uses raw diff and chi_square is 0.
        flux_estimator = FluxEstimator(None)
        mean_fluxes = np.array([[1.0, 0.0, 0.0],
                                [4.0, 0.0, 0.0],
                                [7.0, 0.0, 0.0]])
        errors = np.zeros_like(mean_fluxes)
        result = flux_estimator.match_fluxes(mean_fluxes, errors, (4.2, 0, 0))
        assert result['profile'] == 1
        assert result['T'] == 4.0
        assert result['err'] == 0.0
        assert result['chi_square'] == 0
        assert result['reduced_chi_square'] == 0


def _make_pmat_factory(nprofiles, ncomp, rhs_value=0.0):
    """Return a `(factory, pmat)` pair. The pmat mirrors PmatTotTransient:
    backward() returns shape (nprofiles, 1, 1, ncomp), which broadcasts cleanly
    into both estimate()'s div assignment ((nprofiles, 1, 1, ncomp, ncomp)[..., i])
    and reshapes to (nprofiles, ncomp) for the rhs."""
    pmat = MagicMock(name='PmatStub')
    pmat.forward.side_effect = lambda data, amps, pmul=1: None
    pmat.backward.side_effect = lambda data, pmul=1: np.full(
        (nprofiles, 1, 1, ncomp), rhs_value
    )
    factory = MagicMock(side_effect=lambda tod, profiles, ncomp=ncomp: pmat)
    return factory, pmat


@pytest.fixture
def estimate_tod():
    tod = create_mock_tod()
    tod.data = np.ones((4, 200), dtype=float)
    return tod


class TestFluxEstimatorEstimate:
    def test_estimate_uses_nmat_when_provided(self, estimate_tod):
        ncomp, nprofiles = 3, 2
        factory, _ = _make_pmat_factory(nprofiles, ncomp)
        nmat = MagicMock()
        nmat.ivar.return_value = np.full(4, 0.5)
        fe = FluxEstimator(pmat=factory, nmat=nmat)

        rhs, div = fe.estimate(estimate_tod, [MagicMock(), MagicMock()], ncomp=ncomp)

        nmat.ivar.assert_called_once_with(estimate_tod)
        assert rhs.shape == (nprofiles, ncomp)
        assert div.shape == (nprofiles, ncomp, ncomp)

    def test_estimate_falls_back_to_median_ivar_without_nmat(self, estimate_tod):
        ncomp, nprofiles = 3, 1
        factory, pmat = _make_pmat_factory(nprofiles, ncomp)
        fe = FluxEstimator(pmat=factory, nmat=None)

        rhs, div = fe.estimate(estimate_tod, [MagicMock()], ncomp=ncomp)

        assert rhs.shape == (nprofiles, ncomp)
        assert div.shape == (nprofiles, ncomp, ncomp)
        # forward called once per component when computing the div matrix
        assert pmat.forward.call_count == ncomp

    def test_estimate_individual_loops_per_profile(self, estimate_tod):
        ncomp = 3
        # estimate_individual builds a pmat per profile, each with nprofiles=1.
        factory, _ = _make_pmat_factory(nprofiles=1, ncomp=ncomp)
        fe = FluxEstimator(pmat=factory, nmat=None)
        profiles = [MagicMock(), MagicMock(), MagicMock()]

        rhs, div = fe.estimate_individual(estimate_tod, profiles, ncomp=ncomp)

        assert rhs.shape == (3, ncomp)
        assert div.shape == (3, ncomp, ncomp)
        assert factory.call_count == 3