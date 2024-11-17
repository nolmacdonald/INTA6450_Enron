import numpy as np
import pandas as pd
import sqlite3
from utils.log_config import LoggerConfig
from utils.db_manager import DatabaseManager
import time
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class EmailProcessor:
    def __init__(self, df):
        self.df = df
        self.logger = LoggerConfig(logger_name="EmailProcessor").get_logger()

    def text_extract(self, text):
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text() if pd.notnull(text) else ""

        # Remove URLs and email addresses
        text = re.sub(r"\S+@\S+", "", text)  # Remove email addresses
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        return text

    def text_normalize(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    def process_data(
        self, save_csv_path=None, save_db_path=None, table_name="emails_processed"
    ):

        self.logger.info(f"Pre-processing data...")
        start_proc = time.time()
        # Create a copy of the DataFrame
        emails_df = self.df.copy()

        # Text extraction --------------------------------------------------------------
        self.logger.info(f"Starting text extraction...")
        start_ex = time.time()
        emails_df["processed_text"] = emails_df["text"].apply(self.text_extract)
        end_ex = time.time()
        self.logger.info(f"Performed text extraction in {end_ex - start_ex:.2f} s")

        # Normalize Text ---------------------------------------------------------------
        self.logger.info(f"Starting text normalization...")
        start_norm = time.time()
        emails_df["processed_text"] = emails_df["processed_text"].apply(
            self.text_normalize
        )
        end_norm = time.time()
        self.logger.info(
            f"Performed text normalization in {end_norm - start_norm:.2f} s"
        )

        # Tokenization -----------------------------------------------------------------
        self.logger.info(f"Starting tokenization...")
        # Tokenization splits the text into individual words (tokens)
        start_token = time.time()
        emails_df["tokens"] = emails_df["processed_text"].apply(word_tokenize)
        end_token = time.time()
        self.logger.info(f"Performed tokenization in {end_token - start_token:.2f} s")

        # Stop Words Removal -----------------------------------------------------------
        self.logger.info(f"Starting stop words removal...")
        # Filter out common stop words that don’t carry significant meaning
        start_stop = time.time()
        stop_words = set(stopwords.words("english"))

        emails_df["tokens"] = emails_df["tokens"].apply(
            lambda x: [word for word in x if word not in stop_words]
        )

        end_stop = time.time()
        self.logger.info(
            f"Performed stop words removal in {end_stop - start_stop:.2f} s"
        )

        # Stemming ---------------------------------------------------------------------
        self.logger.info(f"Starting stemming...")
        # Stemming reduces words to their root form, which helps group similar words.
        stemmer = PorterStemmer()
        start_stem = time.time()
        emails_df["tokens"] = emails_df["tokens"].apply(
            lambda x: [stemmer.stem(word) for word in x]
        )
        end_stem = time.time()
        self.logger.info(f"Performed stemming in {end_stem - start_stem:.2f} s")

        # Joining Tokens ---------------------------------------------------------------
        self.logger.info(f"Starting token joining...")
        # Join the tokens back into sentences for the CountVectorizer to process
        # Then proceed with your LDA modeling
        start_join = time.time()
        emails_df["final_text"] = emails_df["tokens"].apply(lambda x: " ".join(x))
        end_join = time.time()
        self.logger.info(f"Performed token joining in {end_join - start_join:.2f} s")

        end_proc = time.time()
        self.logger.info(f"Pre-processed data in {end_proc - start_proc:.2f} s")

        # Save the DataFrame to a CSV file if requested
        if save_csv_path:
            manager = DatabaseManager(file_path=save_csv_path)
            manager.save_to_csv(emails_df)
            self.logger.info(f"Parsed dataset saved to CSV file: {save_csv_path}")

        # Save the DataFrame to a SQLite database if requested
        if save_db_path:
            manager = DatabaseManager(file_path=save_db_path)
            manager.save_to_db(emails_df, table_name)
            self.logger.info(f"Parsed dataset saved to SQLite database: {save_db_path}")

        return emails_df


if __name__ == "__main__":
    # Connect to the database (or create it if it doesn't exist)
    connection = sqlite3.connect("../emails.db")

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Load the dataframe from the SQLite database
    table_name = "emails"
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    # Close the connection
    connection.close()

    proc = EmailProcessor(df)
    emails_df = proc.process_data(
        save_db_path=f"{os.path.dirname(os.getcwd())}/emails.db"
    )  # Save the DataFrame to SQLite database)

    print(emails_df.head())
