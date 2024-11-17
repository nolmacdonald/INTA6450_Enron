import sqlite3
import time
import os
import pandas as pd


class DatabaseManager:
    def __init__(self, db_path, logger=None):
        self.db_path = db_path
        self.conn = sqlite3.connect(f"{self.db_path}")
        self.cursor = self.conn.cursor()
        self.logger = logger

    def load_db(self, table_name):
        """
        Load all data from the specified table in an SQLite database.

        Parameters:
            table_name (str): The name of the table to fetch data from.

        Returns:
            list: A list of tuples containing all rows from the table.
            list: A list of column names.
        """
        self.cursor.execute(f"SELECT * FROM {table_name}")
        rows = self.cursor.fetchall()
        column_names = [description[0] for description in self.cursor.description]
        return rows, column_names

    def save_to_csv(self, df, save_path):
        """
        Save the DataFrame to a CSV file.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
        """
        save_start_time = time.time()
        df.to_csv(save_path, index=False)
        save_end_time = time.time()
        if self.logger:
            self.logger.info(
                f"Emails saved to CSV in {save_end_time - save_start_time:.2f} seconds."
            )
            self.logger.info(f"Parsed emails saved at {save_path}")

    def save_to_db(self, df, table_name):
        """
        Save the DataFrame to a SQLite database.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
            table_name (str): The name of the table to save the data to.
        """
        save_start_time = time.time()
        conn = sqlite3.connect(f"{self.db_path}")

        # Convert list columns to strings before saving to SQLite
        for column in df.columns:
            if df[column].apply(type).eq(list).any():
                df[column] = df[column].apply(str)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        save_end_time = time.time()
        if self.logger:
            self.logger.info(
                f"Emails saved to SQLite database in {save_end_time - save_start_time:.2f} seconds."
            )
            self.logger.info(f"Parsed emails saved at {self.db_path}")

    @staticmethod
    def ensure_directory_exists(directory, logger=None):
        """
        Ensure that the specified directory exists. If it does not exist, create it.

        Parameters:
            directory (str): The path to the directory.
            logger (logging.Logger): The logger instance for logging.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            if logger:
                logger.info(f"Directory created: {directory}")
        else:
            if logger:
                logger.info(f"Directory already exists: {directory}")

    def close_db(self):
        """Close the cursor and connection."""
        self.cursor.close()
        self.conn.close()


# Example usage
if __name__ == "__main__":
    # Get the absolute path of the current directory (e.g., src/utils)
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Navigate up two levels to reach the root directory
    # from src/utils root dir is down two levels to INTA6450_Enron/
    root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

    # Initialize the DatabaseManager with the path to the SQLite database
    db_manager = DatabaseManager(db_path=f"{root_dir}/data/emails.db")

    # Load data from the "emails" table in the SQLite database
    rows, column_names = db_manager.load_db(table_name="emails")
    print(f"Column names: {column_names}")

    # Close the database connection
    db_manager.close_db()
