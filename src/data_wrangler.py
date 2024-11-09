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
        self.data_saver = DatabaseManager(self.logger)

    def parse_emails(self, save_csv_path=None, save_db_path=None, table_name="emails"):
        """
        Load emails from JSON files located in the specified directory and return them as a pandas DataFrame.

        This method reads all JSON files in the directory specified by `self.json_dir`, extracts relevant email
        information from each file, and compiles the data into a pandas DataFrame. The method also logs the
        progress and time taken to process each file and the overall operation.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded email data with the following columns:
            - message_id: The unique identifier of the email.
            - date: The date the email was sent.
            - from: The email address of the sender.
            - from_name: The name of the sender.
            - to: The email addresses of the recipients.
            - to_name: The names of the recipients.
            - cc: The email addresses of the CC recipients.
            - cc_name: The names of the CC recipients.
            - bcc: The email addresses of the BCC recipients.
            - bcc_name: The names of the BCC recipients.
            - subject: The subject of the email.
            - text: The body text of the email.
            - folder: The folder where the email is stored.
            - origin: The origin of the email.
            - filename: The filename of the email.

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
                        "message_id": data["headers"].get("message-id", ""),
                        "date": data["headers"].get("date", ""),
                        "from": data["headers"].get("from", ""),
                        "from_name": data["headers"].get("x-from", ""),
                        "to": data["headers"].get("to", ""),
                        "to_name": data["headers"].get("x-to", ""),
                        "cc": data["headers"].get("cc", ""),
                        "cc_name": data["headers"].get("x-cc", ""),
                        "bcc": data["headers"].get("bcc", ""),
                        "bcc_name": data["headers"].get("x-bcc", ""),
                        "subject": data["headers"].get("subject", ""),
                        "text": data.get("text", ""),
                        "folder": data["headers"].get("x-folder", ""),
                        "origin": data["headers"].get("x-origin", ""),
                        "filename": data["headers"].get("x-filename", ""),
                    }
                    data_list.append(email_data)  # Add the email data to the list

                file_end_time = (
                    time.time()
                )  # Record the end time of processing the file
                self.logger.info(
                    f"Finished processing file: {file_path} in {file_end_time - file_start_time:.6f} seconds"
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
            manager = DatabaseManager(file_path=save_csv_path)
            manager.save_to_csv(emails_df)

        # Save the DataFrame to a SQLite database if requested
        if save_db_path:
            manager = DatabaseManager(file_path=save_db_path)
            manager.save_to_db(emails_df, table_name)

        return emails_df  # Return the DataFrame


if __name__ == "__main__":
    json_dir = os.path.join(os.path.dirname(__file__), "../data/emails")
    data_wrangler = DataWrangler(json_dir)

    emails_df = data_wrangler.parse_emails(
        save_csv_path="data/emails.csv", save_db_path="data/emails.db"
    )
    print(emails_df.head())
