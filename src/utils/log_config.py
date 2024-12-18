# Utilities for the Enron Project
import logging
import os
import time
import logging.handlers

from utils.db_manager import DatabaseManager


class LoggerConfig:
    """
    A configuration class for setting up a logger with specified parameters.

    Attributes:
        log_dir (str): The directory where log files will be stored. Defaults to "./".
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        logger_name (str): The name of the logger. Defaults to "logger".
        logger (logging.Logger): The logger instance configured with the specified parameters.

    Methods:
        configure_logger():
            Configures the logger with a file handler and a stream handler.
            Creates the log directory if it does not exist.
            Sets the logging format and date format.

        get_logger():
            Returns the configured logger instance.
    """

    def __init__(self, log_level=logging.INFO, logger_name="logger"):
        """
        Initialize the LoggerConfig with directory, log level, and logger name.

        Args:
            log_dir (str): The directory where log files will be stored. Default is "./".
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
            logger_name (str): The name of the logger. Default is "logger".
        """
        # Get the absolute path of the current directory (e.g., src/utils)
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        # Navigate up two levels to reach the root directory
        self.root_dir = os.path.abspath(os.path.join(self.current_dir, "../../"))

        # Initialize the LoggerConfig with directory, log level, and logger name
        self.log_dir = f"{self.root_dir}/logs"
        self.log_level = log_level
        self.logger_name = logger_name
        # Create a logger instance with the specified name
        self.logger = logging.getLogger(self.logger_name)
        # Ensure the log directory exists
        DatabaseManager.ensure_directory_exists(self.log_dir, self.logger)
        # Configure the logger
        self.configure_logger()

    def configure_logger(self):
        """
        Configures the logger for the application.

        This method sets up the logging configuration for the application. It ensures that the log directory exists,
        defines the log file path, and configures the logging settings to log messages to both a file and the console.

        Attributes:
            self.log_dir (str): The directory where log files will be stored.
            self.logger_name (str): The name of the logger.
            self.log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

        Creates:
            The log directory if it does not exist.

        Logs:
            Messages to a file named after the logger and to the console with a specific format and date format.
        """
        # Create the log directory if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Define the log file path all lower case
        log_file = os.path.join(
            self.log_dir,
            f"{self.logger_name.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.log",
        )

        # Create a memory handler with a buffer size of 1000 log records
        memory_handler = logging.handlers.MemoryHandler(
            capacity=1000,
            flushLevel=logging.ERROR,
            target=logging.FileHandler(log_file),
        )

        # Configure the logging settings with the memory handler
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[memory_handler, logging.StreamHandler()],
        )

    def flush_memory_handler(self):
        """
        Flush the memory handler to the file handler if there are log messages.
        """
        for handler in self.logger.handlers:
            if isinstance(handler, logging.handlers.MemoryHandler):
                handler.flush()

    def get_logger(self):
        """
        Retrieve the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Flush the memory handler before returning the logger
        self.flush_memory_handler()
        # Return the configured logger instance
        return self.logger
