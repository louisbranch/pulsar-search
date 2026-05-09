from typing import Optional
from logging import WARNING

from .instrument import Instrument
from . import log

class Config:
    """
    Holds the active instrument and log level for the package. A single global
    instance lets module-level helpers (`tods()`, `create_map()`, …) dispatch
    to the active instrument without threading it through every call.

    On construction, falls back to ACT if no instrument is provided and the
    optional ACT extra is installed; otherwise the instrument is None and most
    helpers will fail until `init()` supplies one.
    """
    instrument: Instrument = None

    def __init__(self, instrument: Optional[Instrument] = None, log_level = WARNING):
        log.init(log_level)

        self.instrument = instrument
        if instrument is None:
            try:
                from .act import ACT
                self.instrument = ACT()
            except ImportError as e:
                log.error(f'Failed to import ACT instrument: {e}')
                log.warning("No instrument provided. Most functions will not work.")

# Global config instance
config = Config()

def init(instrument: Optional[Instrument] = None, log_level = WARNING):
    """
    Override the global config instance with the given instrument and log level.
    """
    global config

    # Update the global config instance
    config = Config(instrument, log_level)

def instrument() -> Instrument:
    """
    Returns the current instrument object.
    """
    return config.instrument