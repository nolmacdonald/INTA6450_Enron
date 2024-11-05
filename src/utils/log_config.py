# Utilities for the Enron Project
import logging
import os
import time

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

    def __init__(self, log_dir="./logs", log_level=logging.INFO, logger_name="logger"):
        """
        Initialize the LoggerConfig with directory, log level, and logger name.

        Args:
            log_dir (str): The directory where log files will be stored. Default is "./".
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
            logger_name (str): The name of the logger. Default is "logger".
        """
        # Initialize the LoggerConfig with directory, log level, and logger name
        self.log_dir = log_dir
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

        # Configure the logging settings
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def get_logger(self):
        """
        Retrieve the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Return the configured logger instance
        return self.logger
