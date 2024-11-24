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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class EmailTopics:
    def __init__(self):
        self.logger = LoggerConfig(logger_name="EmailTopics").get_logger()
        self.data_saver = DatabaseManager(self.logger)

    def load_data(self, db_path, table_name):
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        connection.close()
        return emails_df

    def text_extract(self, text):
        text = BeautifulSoup(text, "html.parser").get_text() if pd.notnull(text) else ""
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"http\S+|www\S+", "", text)
        return text

    def text_normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text

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

        # Save the DataFrame to a CSV file if requested
        if save_csv_path:
            manager = DatabaseManager(db_path=save_csv_path)
            manager.save_to_csv(emails_df)
            self.logger.info(f"Parsed dataset saved to CSV file: {save_csv_path}")

        # Save the DataFrame to a SQLite database if requested
        if save_db_path:
            manager = DatabaseManager(file_path=save_db_path)
            manager.save_to_db(emails_df, table_name)
            self.logger.info(f"Parsed dataset saved to SQLite database: {save_db_path}")

        end_proc = time.time()
        self.logger.info(f"Completed pre-processing in {end_proc - start_proc:.2f} s")

        return emails_df

    def get_top_topics(self, model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx}:")
            print(
                " ".join(
                    [
                        feature_names[i]
                        for i in topic.argsort()[: -no_top_words - 1 : -1]
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
    topics = EmailTopics()
    df = topics.process_data(
        emails_df, save_db_path=f"{root_dir}/data/emails_processed.db"
    )

    # Join the tokens back into sentences for the CountVectorizer to process
    # Then proceed with your LDA modeling
    df["final_text"] = df["tokens"].apply(lambda x: " ".join(x))

    # Now, use 'final_text' for LDA
    text_data = df["final_text"].values
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(text_data)

    # LDA
    lda = LatentDirichletAllocation(n_components=20, random_state=42)
    lda.fit(dtm)

    no_top_words = 10
    tf_feature_names = vectorizer.get_feature_names_out()
    top_topics = topics.get_top_topics(lda, tf_feature_names, no_top_words)
    print(top_topics)
