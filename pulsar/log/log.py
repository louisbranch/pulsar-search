import logging
from datetime import datetime, timezone
import sys

class Log:
    """
    Log is a class that provides logging capabilities with variable levels of severity.

    Methods:
    - print: Logs messages with a specified log level if the log level is greater than 
             or equal to the Log object's log level.

    Attributes:
    - logger (logging.Logger): The logger instance from the logging package.
    """

    def __init__(self, level):
        """
        Initializes a Log object with a specified minimum log level.

        Parameters:
        - level: The minimum level of messages that this logger will log.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s UTC] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            formatter.converter = self._utc_time
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def print(self, level, message: str):
        """
        Logs a message with the specified log level if it meets or exceeds the Log's set log level.

        Parameters:
        - level: The severity level of the log message.
        - message (str): The log message to be logged.
        """
        self.logger.log(level, message)

    def _utc_time(self, *args):
        """
        Returns the current UTC time for logging timestamps.

        Returns:
        - time tuple: The current UTC time tuple.
        """
        return datetime.now(timezone.utc).timetuple()

# Global log instance
log = Log(logging.WARNING)

def init(level):
    """
    Initializes the global log instance with the specified log level.

    Parameters:
    - level: The minimum level of messages that the global logger will log.
    """
    global log
    log = Log(level)

def error(message: str):
    """
    Logs an error message using the global log instance.

    Parameters:
    - message (str): The error message to be logged.
    """
    log.print(logging.ERROR, message)

def critical(message: str):
    """
    Logs a critical message using the global log instance and exits the program.

    Parameters:
    - message (str): The critical message to be logged.
    """
    log.print(logging.CRITICAL, message)
    sys.exit(1)

def info(message: str):
    """
    Logs an informational message using the global log instance.

    Parameters:
    - message (str): The informational message to be logged.
    """
    log.print(logging.INFO, message)

def debug(message: str):
    """
    Logs a debug message using the global log instance.

    Parameters:
    - message (str): The debug message to be logged.
    """
    log.print(logging.DEBUG, message)

def warning(message: str):
    """
    Logs a warning message using the global log instance.

    Parameters:
    - message (str): The warning message to be logged.
    """
    log.print(logging.WARNING, message)
