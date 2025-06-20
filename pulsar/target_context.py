import tomli
import numpy as np

from .target import Target
from .filters import FilterOptions
from .timing import PeriodTimingModel, BarycentricTimingModel

class TargetContext:
    """
    A configuration wrapper that encapsulates all contextual information
    needed to initialize a Target, its associated timing model, filter parameters,
    and any observational queries.

    This allows clean separation between domain-specific objects (Target, TimingModel)
    and runtime configuration logic (loading from TOML files).
    """
    def __init__(self, config_path: str, root_dir: str = ""):
        """
        Load configuration and initialize context components.

        Parameters:
            config_path (str): Path to a TOML configuration file.
        """
        with open(config_path, "rb") as f:
            cfg = tomli.load(f)

        tcfg = cfg["target"]
        self.target = Target(
            name=tcfg["name"],
            ra=np.deg2rad(tcfg["ra"]),
            dec=np.deg2rad(tcfg["dec"]),
            amp=tcfg["amp"],
            T=tcfg["T"],
            phi0=tcfg["phi0"],
            D=tcfg["D"],
            radius=np.deg2rad(tcfg["R"]),
        )

        if "timing" in cfg:
            self.timing_model = BarycentricTimingModel(
                target=self.target,
                solution_file=root_dir + cfg["timing"]["solution_file"],
                ephem=cfg["timing"]["ephem"],
            )
        else:
            self.timing_model = PeriodTimingModel(self.target)

        filter = cfg.get("filter", {})
        self.filter = FilterOptions(
            method=filter.get("method", "butterworth"),
            cutoff=filter.get("cutoff", 0),
            order=filter.get("order", 0),
            fknee=filter.get("fknee", 0),
            alpha=filter.get("alpha", 0),
        )

    def __str__(self):
        return f"TargetContext(target={self.target.name}, timing_model={self.timing_model}, query_len={len(self.query)})"