from dataclasses import dataclass
from typing import List, Optional

import h5py
import numpy as np
import numpy.ma as ma

from .filters import filter, FilterOptions
from .target import Target, Source
from .tod import TOD
from .io import tods
from . import log

@dataclass
class FluxStat:
    mean: np.array
    median: np.array
    std: np.array
    min: np.array
    max: np.array

@dataclass
class FluxStatResult:
    overall: FluxStat
    target: Optional[FluxStat] = None
    background: Optional[FluxStat] = None

class FluxStats:

    def create(self, filename: str, tod_path: str, opt: FilterOptions,
             target: Optional[Target] = None, sources: Optional[List[Source]] = None, tod_limit: int = 0):
        """
        Create a HDF5 file containing the flux statistics for the TODs in the given path.
        The statistics include mean, median, standard deviation, minimum, and maximum flux values,
        and are calculated for the overall flux, target flux, and background flux (if target is provided).

        Parameters:
            filename (str): The path to the output HDF5 file.
            tod_path (str): The path to the TOD files.
            opt (FilterOptions): The filter options to apply to the TOD.
            target (Optional[Target]): The target object specifying the coordinates of the pulsar region.
            sources (Optional[List[Source]]): A list of source objects specifying the coordinates of other source regions.
            tod_limit (int): The maximum number of TODs to process. If 0, process all TODs in the path.
        """

        for tod in tods(tod_path, limit=tod_limit):
            tod.calibrate()
            if sources:
                # FIXME: This is a temporary solution to apply the filter to multiple sources
                for source in sources:
                    filter(tod, opt, source)
            else:
                filter(tod, opt)
            stats = self._calculate_flux_statistics(tod, target)
            self._save_flux_statistics_to_hdf5(filename, tod.id, stats)

    def _calculate_flux_statistics(self, tod: TOD, target: Optional[Target] = None) -> FluxStatResult:
        """
        Calculate statistical measures of the flux on the TOD, including mean, median, 
        standard deviation, minimum, and maximum flux values. Optionally, calculate 
        these statistics for the flux values within a specified target range and
        for the flux values excluding this target range (background).

        Parameters:
            tod (TOD): The time-ordered data object containing the data and scan information.
            target (Optional[Target]): The target object specifying the coordinates of the region.

        Returns:
            FluxStatResult: An object containing the calculated statistics for overall, 
                                  target, and background fluxes.
        """
        
        def stats(data):
            return FluxStat(
                mean=np.mean(data, axis=1),
                median=np.median(data, axis=1),
                std=np.std(data, axis=1),
                min=np.min(data, axis=1),
                max=np.max(data, axis=1)
            )

        overall = stats(tod.data)

        if target is None:
            return FluxStatResult(overall=overall)

        _, samples = tod.locate_source(target.ra, target.dec, target.radius)
        mask = samples.to_mask()

        t = stats(ma.array(tod.data, mask=~mask))
        bg = stats(ma.array(tod.data, mask=mask))

        log.info(f'TOD: {tod.id}: Overall: {np.mean(overall.mean)}, Target: {np.mean(t.mean)}, Background: {np.mean(bg.mean)}')

        return FluxStatResult(overall=overall, target=t, background=bg)

    def _save_statistics_group(self, group: h5py.Group, stats: FluxStat):
        group.create_dataset("mean", data=stats.mean)
        group.create_dataset("median", data=stats.median)
        group.create_dataset("std", data=stats.std)
        group.create_dataset("min", data=stats.min)
        group.create_dataset("max", data=stats.max)

    def _save_flux_statistics_to_hdf5(self, filename: str, tod_id: str, stats: FluxStatResult):
        with h5py.File(filename, 'a') as f:
            tod_group = f.create_group(f"{tod_id}")

            overall_group = tod_group.create_group("overall")
            self._save_statistics_group(overall_group, stats.overall)

            if stats.target:
                target_group = tod_group.create_group("target")
                self._save_statistics_group(target_group, stats.target)

            if stats.background:
                background_group = tod_group.create_group("background")
                self._save_statistics_group(background_group, stats.background)

    def _read_statistics_group(self, group: h5py.Group) -> FluxStat:
        return FluxStat(
            mean=group["mean"][()],
            median=group["median"][()],
            std=group["std"][()],
            min=group["min"][()],
            max=group["max"][()]
        )

    def _calculate_average_statistics(self, stats_list: List[FluxStat]) -> FluxStat:
        mean = np.mean(np.concatenate([s.mean for s in stats_list]))
        median = np.mean(np.concatenate([s.median for s in stats_list]))
        std = np.mean(np.concatenate([s.std for s in stats_list]))
        min = np.mean(np.concatenate([s.min for s in stats_list]))
        max = np.mean(np.concatenate([s.max for s in stats_list]))
        
        return FluxStat(
            mean=mean,
            median=median,
            std=std,
            min=min,
            max=max
        )

    def read(self, filename: str) -> FluxStatResult:
        """
        Read the flux statistics from the HDF5 file and calculate the average statistics for the TODs.
        """

        overall_stats_list, target_stats_list, background_stats_list = [], [], []

        with h5py.File(filename, 'r') as f:
            for tod_id in f.keys():
                tod_group = f[tod_id]

                overall_stats = self._read_statistics_group(tod_group["overall"])
                overall_stats_list.append(overall_stats)

                if "target" in tod_group:
                    target_stats = self._read_statistics_group(tod_group["target"])
                    target_stats_list.append(target_stats)

                if "background" in tod_group:
                    background_stats = self._read_statistics_group(tod_group["background"])
                    background_stats_list.append(background_stats)

        overall_avg = self._calculate_average_statistics(overall_stats_list)
        
        if len(target_stats_list) > 0:
            target_avg = self._calculate_average_statistics(target_stats_list)
        else:
            target_avg = None

        if len(background_stats_list) > 0:
            background_avg = self._calculate_average_statistics(background_stats_list)
        else:
            background_avg = None

        return FluxStatResult(
            overall=overall_avg,
            target=target_avg,
            background=background_avg
        )

def pad_with_nan(array, target_shape):
    result = np.full(target_shape, np.nan)
    slices = tuple(slice(0, min(dim, target)) for dim, target in zip(array.shape, target_shape))
    result[slices] = array[slices]
    return result

def compute_with_method(arrays, method):
    # Determine the target shape (maximum shape in each dimension)
    target_shape = tuple(max(arr.shape[dim] for arr in arrays) for dim in range(arrays[0].ndim))
    
    # Pad each array with NaNs
    padded_arrays = [pad_with_nan(arr, target_shape) for arr in arrays]
    
    # Stack arrays
    stacked_arrays = np.stack(padded_arrays)
    
    # Calculate the result using the specified method, ignoring NaNs
    if method == np.mean:
        return np.nanmean(stacked_arrays, axis=0)
    elif method == np.median:
        return np.nanmedian(stacked_arrays, axis=0)
    elif method == np.nanmax:
        return np.nanmax(stacked_arrays, axis=0)
    elif method == np.nanmin:
        return np.nanmin(stacked_arrays, axis=0)

    raise ValueError("Unsupported method. Use np.mean, np.nanmax, or np.nanmin.")