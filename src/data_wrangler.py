import os
import json
import time
import pandas as pd
from utils.log_config import LoggerConfig
from utils.db_manager import DatabaseManager


class DataWrangler:
    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.logger = LoggerConfig(logger_name="DataWrangler").get_logger()
        current_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.abspath(os.path.join(current_dir, "../data/emails.db"))
        self.data_saver = DatabaseManager(db_path,self.logger)

    def parse_emails(self, save_csv_path=None, save_db_path=None, table_name="emails"):
        """
        Load emails from JSON files located in the specified directory and return them
        as a pandas DataFrame.

        This method reads all JSON files in the directory specified by `self.json_dir`,
        extracts relevant email information from each file, and compiles the data into
        a pandas DataFrame. The method also logs the progress and time taken to process
        each file and the overall operation.

        Returns:
            pd.DataFrame: A DataFrame containing email data with the following columns:
            - text: The body text of the email.
            - message_id: The unique identifier of the email.
            - date: The date the email was sent.
            - from: The email address of the sender.
            - to: The email addresses of the recipients.
            - subject: The subject of the email.
            - cc: The email addresses of the CC recipients.
            - bcc: The email addresses of the BCC recipients.
            - mime-version: The MIME version of the email.
            - content-type: The content type of the email.
            - content-transfer-encoding: The content transfer encoding of the email.
            - x-from: The name of the sender.
            - x-to: Names of the recipients.
            - x-cc: Names of the CC recipients.
            - x-bcc: Names of the BCC recipients.
            - folder: The folder where the email is stored.
            - origin: The origin of the email.
            - filename: The filename of the email.
            - priority: The priority of the email.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
            JSONDecodeError: If any of the JSON files cannot be decoded.
        """
        start_time = time.time()  # Record the start time of the loading process
        self.logger.info(
            "Loading emails from JSON files..."
        )  # Log the start of the loading process
        data_list = []  # Initialize an empty list to store email data

        # Iterate over all files in the specified directory
        for filename in os.listdir(self.json_dir):
            if filename.endswith(".json"):  # Process only JSON files
                file_path = os.path.join(
                    self.json_dir, filename
                )  # Get the full path of the file
                self.logger.info(
                    f"Processing file: {file_path}"
                )  # Log the file being processed
                file_start_time = (
                    time.time()
                )  # Record the start time of processing the file

                with open(file_path, "r") as file:  # Open the JSON file
                    data = json.load(file)  # Load the JSON data
                    # Extract relevant fields from the JSON data
                    email_data = {
                        # Main email data
                        "text": data.get("text", ""),
                        # Headers
                        "message_id": data["headers"].get("message-id", ""),
                        "date": data["headers"].get("date", ""),
                        "from": data["headers"].get("from", ""),
                        "to": data["headers"].get("to", ""),
                        "subject": data["headers"].get("subject", ""),
                        "cc": data["headers"].get("cc", ""),
                        "bcc": data["headers"].get("bcc", ""),
                        "mime-version": data["headers"].get("mime-version", ""),
                        "content-type": data["headers"].get("content-type", ""),
                        "content-transfer-encoding": data["headers"].get("", ""),
                        "x-from": data["headers"].get("x-from", ""),
                        "x-to": data["headers"].get("x-to", ""),
                        "x-cc": data["headers"].get("x-cc", ""),
                        "x-bcc": data["headers"].get("x-bcc", ""),
                        "folder": data["headers"].get("x-folder", ""),
                        "origin": data["headers"].get("x-origin", ""),
                        "filename": data["headers"].get("x-filename", ""),
                        # Main email data
                        # Commented out since these are duplicate information
                        # "subject": data.get("subject", ""),
                        # "messageId": data.get("messageId", ""),
                        # "date": data.get("date", ""),
                        # "from": data.get("from", ""),
                        # "to": data.get("to", ""),
                        # "cc": data.get("cc", ""),
                        # "bcc": data.get("bcc", ""),
                        # "date": data.get("date", ""),
                        "priority": data.get("priority", ""),
                    }
                    data_list.append(email_data)  # Add the email data to the list

                file_end_time = (
                    time.time()
                )  # Record the end time of processing the file

                # Log
                # Total time to parse emails
                log_proc = f"{file_end_time - file_start_time:.6f} s"
                self.logger.info(
                    f"Finished processing file: {file_path} in {log_proc}"
                )  # Log the time taken to process the file

        # Convert the list of email data to a DataFrame
        emails_df = pd.DataFrame(data_list)

        # Record the end time of the loading process
        end_time = time.time()

        # Log the successful loading of emails and the time taken
        self.logger.info(
            f"Emails loaded successfully in {end_time - start_time:.2f} seconds."
        )

        # Save the DataFrame to a CSV file if requested
        if save_csv_path:
            manager = DatabaseManager(db_path=save_csv_path)
            manager.save_to_csv(emails_df)
            self.logger.info(f"Parsed dataset saved to CSV file: {save_csv_path}")

        # Save the DataFrame to a SQLite database if requested
        if save_db_path:
            manager = DatabaseManager(db_path=save_db_path)
            manager.save_to_db(emails_df, table_name)
            self.logger.info(f"Parsed dataset saved to SQLite database: {save_db_path}")

        return emails_df  # Return the DataFrame


if __name__ == "__main__":
    # Get the absolute path of the current directory (e.g., src/utils)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # Navigate up one level to reach the root directory
    root_dir = os.path.abspath(os.path.join(current_dir, "../"))

    path_emails_raw = f"{root_dir}/data/emails/"
    data_wrangler = DataWrangler(path_emails_raw)

    # Parse the emails by reading all json files and recording to DataFrame
    # Save to a CSV file: save_csv_path=f"{root_dir}/path/to/file.csv"
    # Save to a SQLite3 database: save_db_path=f"{root_dir}/path/to/database.db"
    emails_df = data_wrangler.parse_emails(
        # os.path.dirname(os.getcwd()) is root dir - INTA6450_Enron/ folder
        save_db_path=f"{root_dir}/data/emails.db",
    )
    print(emails_df.head())
