from typing import List, Tuple

import numpy as np
import pandas as pd
from pixell import utils

from .tod import TOD
from .pointing_models import PointingModel
from .signal_profile import SignalProfile
from . import log

class FluxEstimator:
    """
    FluxEstimator class to estimate the flux of a source in the TODs.
    The class takes a list of profiles, and calculates the flux of the source in a TOD.

    Attributes:
    -----------
    pmat (PointingModel): The pointing model object which maps sky signals into TOD.
    nmat (Nmat): The noise matrix object which models the noise characteristics of the TOD.
    """
    def __init__(self, pmat: PointingModel, nmat: object = None):
        """
        Initializes the FluxEstimator with provided Pmat and Nmat objects.

        Parameters:
            pmat (PointingModel): The pointing model object which maps sky signals into TOD.
            nmat (Nmat): The noise matrix object which models the noise characteristics of the TOD.
        """
        self.pmat: PointingModel = pmat
        self.nmat = nmat

    def estimate(self, tod: TOD, profiles: List[SignalProfile],
                 ncomp: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the per-TOD sufficient statistics (rhs, div) for a linear flux
        estimate. Returning the pieces — rather than a flux value — lets the
        search sum rhs/div over many TODs and invert once, which is both cheaper
        and statistically correct for the joint estimate.

        All profiles share a single pointing matrix solve. Use
        `estimate_individual` for phase-exclusive profiles whose pulses overlap.

        Parameters:
            tod (TOD): The time-ordered object containing the data and scan information.
            profiles (List[Profile]): The list of profiles to be used for the flux estimation.
            ncomp (int): The number of polarization components to use for the flux estimation.
                Default is 3: [T, Q, U].

        Returns:
            tuple: A tuple containing:
                   - rhs (np.ndarray): The right-hand side T^t N^-1 d.
                   - div (np.ndarray): The normal matrix T^t N^-1 T.
        """

        log.debug(f"Estimating Flux for TOD ID: {tod.id}")

        nprofiles = len(profiles)

        pmat = self.pmat(tod, profiles, ncomp=ncomp)
        
        # Get the inverse variance directly from the noise model
        if self.nmat is not None:
            ivar = self.nmat.ivar(tod)
        else:
            ivar = np.median(utils.block_reduce(tod.data**2, 100, inclusive=False),-1)**-1
        
        # Apply N^-1 to the TOD
        tod.data *= ivar[:, None]  # Scale TOD by the inverse variance

        # Fill gaps if necessary
        tod.fill_gaps()

        # Compute T^t (simulating T^t N^-1 d)
        rhs = pmat.backward(tod.data)

        # Initialize a zero TOD for calculating div (T N^-1 T^t)
        div_shape = [nprofiles, 1, 1, ncomp, ncomp]
        div = np.zeros(div_shape)
        
        for i in range(ncomp):
            zero_tod = np.zeros_like(tod.data)
            amps = np.zeros(rhs.shape)
            amps[..., i] = 1

            # T
            pmat.forward(zero_tod, amps)

            # Apply inverse variance (N^-1)
            zero_tod *= ivar[:, None]
            tod.fill_gaps(zero_tod)

            # T^t
            div[..., i] = pmat.backward(zero_tod)
        
        # Reshape based on ncomp
        rhs = rhs.reshape(nprofiles, ncomp)
        div = div.reshape(nprofiles, ncomp, ncomp)

        return rhs, div

    def estimate_individual(self, tod: TOD, profiles: List[SignalProfile],
                            ncomp: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same statistics as `estimate`, but solves each profile in its own
        pointing matrix. Required for phase-exclusive profiles (e.g. von Mises)
        whose pulses extend beyond a single phase bin and would interfere if
        solved jointly. See `SignalProfile.phase_exclusive`.

        Parameters and return values match `estimate`.
        """

        nprofiles = len(profiles)

        # Initialize arrays for the combined rhs and div results
        rhs = np.zeros((nprofiles, ncomp))
        div = np.zeros((nprofiles, ncomp, ncomp))

        # Get the inverse variance directly from the noise model
        if self.nmat is not None:
            ivar = self.nmat.ivar(tod)
        else:
            ivar = np.median(utils.block_reduce(tod.data**2, 100, inclusive=False),-1)**-1

        # Apply N^-1 to the TOD
        tod.data *= ivar[:, None]  # Scale TOD by the inverse variance

        # Fill gaps if necessary
        tod.fill_gaps()

        # Loop over each profile individually
        for iprofile, profile in enumerate(profiles):
            
            # Step 1: Construct pmat for this profile only
            pmat = self.pmat(tod, [profile], ncomp=ncomp)
            
            # Step 2: Calculate rhs for this profile (T^t N^-1 d)
            profile_rhs = pmat.backward(tod.data)  # Shape: (ncomp,)
            rhs[iprofile] = profile_rhs

            # Step 3: Calculate div for this profile (T N^-1 T^t)
            div_shape = [ncomp, ncomp]
            profile_div = np.zeros(div_shape)
            
            for i in range(ncomp):
                zero_tod = np.zeros_like(tod.data)
                amps = np.zeros(profile_rhs.shape)
                amps[..., i] = 1

                # T forward
                pmat.forward(zero_tod, amps)

                # Apply inverse variance (N^-1)
                zero_tod *= ivar[:, None]
                tod.fill_gaps(zero_tod)

                # T backward
                profile_div[..., i] = pmat.backward(zero_tod)
            
            # Step 4: Store the div for this profile
            div[iprofile] = profile_div

        # Return combined rhs and div results
        return rhs, div

    def calculate_flux(self, data, nsplits: int = 20, subtract_median_flux: bool = True):
        """
        Calculate flux and signal-to-noise ratio (SNR) by splitting the input data into subsets.

        Each subset (split) groups a subset of TODs (time-ordered data) to estimate variance across the dataset.
        Flux is calculated by solving the linear system: flux = inv(div) · rhs

        Parameters:
        - data (np.ndarray): Structured array with fields 'rhs' and 'div' per TOD.
                            - rhs shape: (n_profiles, n_components)      — usually (N, 3)
                            - div shape: (n_profiles, n_components, n_components) — usually (N, 3, 3)
        - nsplits (int): Number of subsets to divide TODs into. Used to estimate statistical uncertainty.
        - subtract_median_flux (bool): If True, subtracts the per-split median flux (per profile/component) to remove baseline offsets.

        Returns:
        - raw_fluxes (np.ndarray): Flux per split before median subtraction. Shape: (nsplits, n_profiles, n_components)
        - fluxes (np.ndarray): Flux per split after median subtraction (if enabled). Same shape as above.
        - mean_flux (np.ndarray): Mean flux across all splits. Shape: (n_profiles, n_components)
        - err (np.ndarray): Standard error (per profile/component). Shape: (n_profiles, n_components)
        - snr (np.ndarray): Signal-to-noise ratio (|mean_flux / err|). Shape: (n_profiles, n_components)
        """

        data.sort(order=['tod_id'])
        tod_ids = np.unique(data['tod_id'])

        if nsplits > len(tod_ids):
            log.error(f'Number of splits {nsplits} is greater than the number of TODs {len(tod_ids)}.')
            raise ValueError('Number of splits must be less than or equal to the number of TODs.')

        rhs_shape = data['rhs'][0].shape  # Typically (n_profiles, n_components) — e.g., (N, 3)
        div_shape = data['div'][0].shape  # Typically (n_profiles, n_components, n_components) — e.g., (N, 3, 3)

        log.debug(f'Calculating flux with {nsplits} splits for rhs {rhs_shape} and div {div_shape}.')

        # Identify and remove TODs with invalid div matrices (NaN or Inf)
        bad_tods = set()
        for tod in tod_ids:
            tod_data = data[data['tod_id'] == tod]
            if np.isnan(tod_data['div']).any() or np.isinf(tod_data['div']).any():
                bad_tods.add(tod)
                log.warning(f"Matrix 'div' contains NaN or Inf values! Removing TOD ID: {tod}")

        # Clean dataset: remove all rows associated with bad TODs
        data_clean = data[~np.isin(data['tod_id'], list(bad_tods))]
        tod_ids = np.unique(data_clean['tod_id'])  # Update list

        # Preallocate arrays for fluxes
        raw_fluxes = np.zeros((nsplits,) + rhs_shape)   # Shape: (nsplits, n_profiles, n_components)
        fluxes = np.zeros((nsplits,) + rhs_shape)

        # Split TODs into nsplits groups (rotating assignment)
        splits = [tod_ids[i::nsplits] for i in range(nsplits)]

        for i, split in enumerate(splits):
            split_data = data_clean[np.isin(data_clean['tod_id'], split)]

            # Sum rhs and div matrices over all TODs in the split
            rhs = split_data['rhs'].sum(axis=0)  # Shape: (n_profiles, n_components)
            div = split_data['div'].sum(axis=0)  # Shape: (n_profiles, n_components, n_components)

            if np.isnan(div).any() or np.isinf(div).any():
                log.warning(f"Unexpected NaN/Inf in 'div' after filtering. Split index: {i}")

            # Solve flux = inv(div) · rhs using matrix pseudo-inverse
            idiv = np.linalg.pinv(div)  # Shape: (n_profiles, n_components, n_components)
            raw_flux = np.einsum('iab,ib->ia', idiv, rhs)  # Shape: (n_profiles, n_components)

            raw_fluxes[i] = raw_flux
            fluxes[i] = raw_flux

        # Estimate uncertainty
        if nsplits == 1:
            # Estimate error from inverse sqrt of diagonal of normal matrix
            diagonal_div = np.diagonal(div, axis1=-2, axis2=-1)  # Shape: (n_profiles, n_components)
            diagonal_div = np.where(diagonal_div == 0, 1e-10, diagonal_div)
            err = np.sqrt(1 / diagonal_div)
        else:
            # Empirical standard error from split variation
            err = np.std(fluxes, axis=0) / np.sqrt(nsplits)

        # Compute mean flux and signal-to-noise ratio
        mean_flux = fluxes.mean(axis=0)  # Shape: (n_profiles, n_components)

        if subtract_median_flux:
            # Subtract per-split median (per profile/component) to remove baseline offsets
            # Shape of medians: (nsplits, 1, n_components), broadcasted along time/profile axis
            #medians = np.median(fluxes, axis=1, keepdims=True)
            medians = np.median(mean_flux, axis=0, keepdims=True)
            mean_flux -= medians

        snr = np.zeros_like(mean_flux)
        if not np.all(err == 0):
            snr = np.abs(mean_flux / err)

        return raw_fluxes, fluxes, mean_flux, err, snr


    def extract_tod_data(self, tod_id, dataframes):
        """
        Extract the `rhs` and `div` data for a specific `tod_id` from a list of structured arrays
        and return a pandas DataFrame.

        Parameters:
        - tod_id (str): The `tod_id` to filter.
        - dataframes (list of np.ndarray): List of structured arrays, where the first contains
        the real target data, and the rest contain null location data.

        Returns:
        - pd.DataFrame: DataFrame with columns ['tod_id', 'rhs', 'div', 'source'].
        """
        # Initialize an empty list to store filtered data
        combined_data = []

        for i, data in enumerate(dataframes):
            # Determine source label
            source_type = 'target' if i == 0 else f'null_{i}'

            # Filter for the specific `tod_id`
            filtered_data = data[data['tod_id'] == tod_id]
            if filtered_data.size > 0:
                # Convert filtered data to a list of dictionaries for easy DataFrame conversion
                for _, entry in filtered_data.iterrows():
                    combined_data.append({
                        'tod_id': entry['tod_id'],
                        'rhs': entry['rhs'],
                        'div': entry['div'],
                        'source': source_type
                    })

        # Convert combined data to a DataFrame
        if combined_data:
            return pd.DataFrame(combined_data)
        else:
            # Return an empty DataFrame with the expected columns if no data is found
            return pd.DataFrame(columns=['tod_id', 'rhs', 'div', 'source'])

    def match_fluxes(self, mean_fluxes, errors, target):
        """
        Find the best match for target_T given fluxes and their error bars.
        
        Parameters:
        mean_fluxes (np.ndarray): Array of fluxes with shape (nprofiles, 1) or (nprofiles, 3).
        errors (np.ndarray): Array of error bars with the same shape as fluxes.
        target (float): The target flux value [T] to match.
        
        Returns:
        dict: A dictionary containing the best match profile (nprofile), the matched T value,
            the associated error, the difference between target and matched T,
            and the chi-square value.
        """
        
        # Extract the T values and their errors
        fluxes = mean_fluxes[:, 0]
        errors = errors[:, 0]
        target = target[0]

        # Calculate the absolute difference between target_T and T_values
        diff = np.abs(fluxes - target)
        
        # Find the indices of the minimum difference, considering the error
        if np.all(errors == 0):
            idx = np.argmin(diff)
        else:
            idx = np.argmin(diff / errors)
        
        # Best match values
        T = fluxes[idx]
        err = errors[idx]

        # Calculate the difference and chi-square value
        diff_value = T - target
        if err == 0:
            chi_square = 0
        else:
            chi_square = (diff_value / err) ** 2

        # Degrees of freedom: number of data points minus 1 (since we're only fitting one parameter)
        dof = mean_fluxes.shape[1] - 1
        reduced_chi_square = chi_square / dof
        
        # Return the results in a dictionary
        result = {
            'profile': idx,
            'T': T,
            'err': err,
            'difference': diff_value,
            'chi_square': chi_square,
            'reduced_chi_square': reduced_chi_square,
        }
        
        return result
