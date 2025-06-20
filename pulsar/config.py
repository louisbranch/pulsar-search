from typing import Optional
from logging import WARNING

from .instrument import Instrument
from . import log

class Config:
    """
    Configure the pulsar module with the instrument and log level.
    The default instrument is ACT and the default log level is WARNING.
    Call init() to update the global config instance.
    Module calls such as `.tods()` will use the global config instance.
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