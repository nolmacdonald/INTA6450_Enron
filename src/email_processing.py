import numpy as np
import pandas as pd
import sqlite3
from utils.log_config import LoggerConfig
from utils.db_manager import DatabaseManager
import time
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime


class EmailProcessing:
    def __init__(self):
        self.logger = LoggerConfig(logger_name="EmailProcessing").get_logger()
        self.data_saver = DatabaseManager(self.logger)

    def load_data(self, db_path, table_name):
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        connection.close()
        return emails_df

    def text_extract(self, text):
        if pd.notnull(text):  # Ensure text is not NaN
            # Use BeautifulSoup only if the text looks like HTML
            if "<" in text and ">" in text:
                text = BeautifulSoup(text, "html.parser").get_text()
            # Remove email addresses
            text = re.sub(r"\S+@\S+", "", text)
            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)
            # Remove new lines and tabs
            text = re.sub(r"\n", "", text)
            text = re.sub(r"\t", "", text)
            text = re.sub(r"\\n", "", text)
            text = re.sub(r"\\t", "", text)

        else:
            text = ""
        return text

    def text_normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    def format_date(self, df):
        # Copy the dataframe to avoid modifying the original data
        emails_df = df.copy()

        # Identify non-matching rows
        non_matching = emails_df[
            ~emails_df["date"].str.match(
                r".*, \d{1,2} \w{3} \d{4} \d{2}:\d{2}:\d{2} [+-]\d{4}"
            )
        ]
        print(non_matching["date"].head())

        # Extract the desired date format
        emails_df["stripped_date"] = emails_df["date"].apply(
            lambda x: (
                re.search(r".{3}, \d{1,2} \w{3} \d{4} \d{2}:\d{2}:\d{2}", x).group(0)
                if re.search(r".{3}, \d{1,2} \w{3} \d{4} \d{2}:\d{2}:\d{2}", x)
                else None
            )
        )

        # Identify rows where extraction failed
        invalid_rows = emails_df[emails_df["stripped_date"].isna()]
        print(f"Invalid rows:\n{invalid_rows}")

        # Parse the stripped dates
        emails_df["datetime"] = emails_df["stripped_date"].apply(
            lambda x: datetime.strptime(x, "%a, %d %b %Y %H:%M:%S")
        )

        return emails_df

    def process_data(
        self, df, save_csv_path=None, save_db_path=None, table_name="emails_processed"
    ):
        self.logger.info(f"Pre-processing data")
        start_proc = time.time()
        emails_df = df.copy()

        # Text extraction
        start_extraction = time.time()
        self.logger.info(f"Starting text extraction")
        emails_df["processed_text"] = emails_df["text"].apply(self.text_extract)
        end_extraction = time.time()
        self.logger.info(
            f"Completed text extraction in {end_extraction - start_extraction:.2f} s"
        )

        # Text normalization
        start_normalization = time.time()
        self.logger.info(f"Starting text normalization")
        emails_df["processed_text"] = emails_df["processed_text"].apply(
            self.text_normalize
        )
        end_normalization = time.time()
        self.logger.info(
            f"Completed normalization in {end_normalization - start_normalization:.2f} s"
        )

        # Tokenization splits the text into individual words (tokens)
        start_token = time.time()
        self.logger.info(f"Starting tokenization")
        emails_df["tokens"] = emails_df["processed_text"].apply(word_tokenize)
        end_token = time.time()
        self.logger.info(f"Completed tokenization in {end_token - start_token:.2f} s")

        # Filter out common stop words that don’t carry significant meaning for topic modeling.
        start_stopwords = time.time()
        stop_words = set(stopwords.words("english"))

        emails_df["tokens"] = emails_df["tokens"].apply(
            lambda x: [word for word in x if word not in stop_words]
        )
        end_stopwords = time.time()
        self.logger.info(
            f"Completed stop word removal in {end_stopwords - start_stopwords:.2f} s"
        )

        # Stemming reduces words to their root form, which helps to group similar words
        start_stem = time.time()
        self.logger.info(f"Starting stemming")
        stemmer = PorterStemmer()
        emails_df["tokens"] = emails_df["tokens"].apply(
            lambda x: [stemmer.stem(word) for word in x]
        )
        end_stem = time.time()
        self.logger.info(f"Completed stemming in {end_stem - start_stem:.2f} s")

        # Format the date to standard date time
        start_date = time.time()
        self.logger.info(f"Formatting date")
        emails_df = self.format_date(emails_df)
        end_date = time.time()
        self.logger.info(f"Completed date formatting in {end_date - start_date:.2f} s")

        # Save the DataFrame to a CSV file if requested
        if save_csv_path:
            manager = DatabaseManager(save_csv_path)
            manager.save_to_csv(emails_df)
            self.logger.info(f"Preprocessed data saved to CSV file: {save_csv_path}")

        # Save the DataFrame to a SQLite database if requested
        if save_db_path:
            manager = DatabaseManager(save_db_path)
            manager.save_to_db(emails_df, table_name)
            self.logger.info(
                f"Preprocessed data saved to SQLite database: {save_db_path}"
            )

        end_proc = time.time()
        self.logger.info(f"Completed preprocessing in {end_proc - start_proc:.2f} s")

        # New information
        self.logger.info(
            f"New columns added to the DataFrame: processed_text, tokens, stripped_date, datetime"
        )

        return emails_df

    def list_top_topics(self, model, feature_names, num_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx}:")
            print(
                " ".join(
                    [
                        feature_names[i]
                        for i in topic.argsort()[: -num_top_words - 1 : -1]
                    ]
                )
            )


if __name__ == "__main__":
    # Define the root directory
    root_dir = "/Users/nmacdonald/projects/INTA6450_Enron"
    # Connect to the database (or create it if it doesn't exist)
    connection = sqlite3.connect(f"{root_dir}/data/emails.db")

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Load the dataframe from the SQLite database
    table_name = "emails"
    emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    # Close the connection
    connection.close()

    # Initialize the EmailTopics class
    email = EmailProcessing()
    df = email.process_data(
        emails_df, save_db_path=f"{root_dir}/data/emails_processed.db"
    )
