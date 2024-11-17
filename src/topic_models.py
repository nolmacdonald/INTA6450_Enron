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


class EmailTopics:
    def __init__(self):
        pass

    # Step 4: Analyze the topics
    def display_topics(self, model, feature_names, no_top_words):
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
    # Connect to the database (or create it if it doesn't exist)
    connection = sqlite3.connect("../emails.db")

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Load the dataframe from the SQLite database
    table_name = "emails_processed"
    emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    # Close the connection
    connection.close()

    # Now, use 'final_text' for LDA
    text_data = emails_df["final_text"].values
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(text_data)

    # LDA
    print(f"Starting LDA...")
    start_lda = time.time()
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(dtm)
    end_lda = time.time()
    print(f"Performed LDA in {end_lda - start_lda:.2f} s")

    # Display topics
    no_top_words = 10
    tf_feature_names = vectorizer.get_feature_names_out()
    EmailTopics().display_topics(lda, tf_feature_names, no_top_words)
