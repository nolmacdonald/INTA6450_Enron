import numpy as np
import pandas as pd
import sqlite3
from utils.log_config import LoggerConfig
from utils.db_manager import DatabaseManager
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os


class TopicModeling:
    def __init__(
        self,
        df,
        num_processors=1,
        save_csv_path=None,
        save_db_path=None,
        topics_table_name="topics",
    ):
        self.logger = LoggerConfig(logger_name="TopicModeling").get_logger()
        self.data_saver = DatabaseManager(self.logger)
        # DataFrame
        self.df = df
        self.df["tokens"] = self.df["tokens"].apply(lambda x: x.split())
        # Number of processors
        self.num_processors = num_processors

        self.save_csv_path = save_csv_path
        self.save_db_path = save_db_path
        self.topics_table_name = topics_table_name

    def create_corpus(self):

        emails_df = self.df.copy()

        # Create a dictionary and a corpus
        dictionary = Dictionary(emails_df["tokens"])
        corpus = [dictionary.doc2bow(tokens) for tokens in emails_df["tokens"]]

        return dictionary, corpus

    def train_lda_model(self, num_passes=10, num_topics=10):
        dictionary, corpus = self.create_corpus()
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=num_passes,
            random_state=42,
            workers=self.num_processors,
        )
        return lda_model

    def coherence_score(self, dictionary, lda_model):

        emails_df = self.df.copy()

        coherence_model_lda = CoherenceModel(
            model=lda_model,
            texts=emails_df["tokens"],
            dictionary=dictionary,
            coherence="c_v",
            processes=self.num_processors,
        )

        # Calculate coherence score
        coherence_score = coherence_model_lda.get_coherence()

        return coherence_score

    def record_dominant_topic(self, lda_model, corpus):
        dominant_topics = []
        for doc_bow in corpus:
            topic_probs = lda_model.get_document_topics(doc_bow)
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            dominant_topics.append(dominant_topic)
        return dominant_topics

    def topic_distribution(self, corpus, lda_model, num_words=10):

        # Step 1: Calculate topic importance (weights)
        topic_importance = [0] * lda_model.num_topics
        for doc in corpus:
            for topic_id, weight in lda_model[doc]:
                topic_importance[topic_id] += weight
        topic_importance = [
            weight / len(corpus) for weight in topic_importance
        ]  # Normalize by number of documents

        # Step 2: Combine importance and terms into a DataFrame
        ranked_topics_data = []

        for topic_id, importance in enumerate(topic_importance):
            topic_terms = lda_model.print_topic(topic_id, topn=num_words)
            ranked_topics_data.append(
                {"Topic": topic_id, "Importance": importance, "Terms": topic_terms}
            )

        # Create DataFrame and sort by importance
        ranked_topics_df = pd.DataFrame(ranked_topics_data).sort_values(
            by="Importance", ascending=False
        )

        # Add new columns for each term and weight
        for i in range(1, num_words + 1):
            ranked_topics_df[f"Term {i}"] = None
            ranked_topics_df[f"Term {i} Weight"] = None

        # Populate the new columns with terms and weights
        for idx, row in ranked_topics_df.iterrows():
            terms_weights = row["Terms"].split(" + ")
            for i, term_weight in enumerate(terms_weights):
                weight, term = term_weight.split("*")
                ranked_topics_df.at[idx, f"Term {i+1}"] = term.strip("'\",")
                ranked_topics_df.at[idx, f"Term {i+1} Weight"] = float(weight)

        ranked_topics_df.reset_index(inplace=True, drop=True)
        return ranked_topics_df

    def topic_model(self, num_passes=10, num_topics=10):
        emails_df = self.df.copy()

        # Create corpus
        dictionary, corpus = self.create_corpus()

        # Train LDA model
        start_training = time.time()
        self.logger.info(
            f"Training LDA model with {num_topics} topics and {num_passes} passes"
        )
        lda_model = self.train_lda_model(num_passes, num_topics)
        end_training = time.time()
        self.logger.info(f"Trained LDA model in {end_training - start_training:.2f} s")

        # Get coherence score
        start_coherence = time.time()
        self.logger.info(f"Calculating coherence score")
        coherence_score = self.coherence_score(dictionary, lda_model)
        self.logger.info(f"Coherence Score: {coherence_score}")
        end_coherence = time.time()
        self.logger.info(
            f"Calculated coherence score in {end_coherence - start_coherence:.2f} s"
        )

        # Record dominant topic number for each email
        self.logger.info(f"Recording dominant topic number to processed email database")
        emails_df["dominant_topic"] = self.record_dominant_topic(lda_model, corpus)

        # Save the DataFrame to a CSV file if requested
        if self.save_csv_path:
            manager = DatabaseManager(self.save_csv_path)
            manager.save_to_csv(emails_df)
            self.logger.info(
                f"Email data and dominant topics saved to CSV file: {save_csv_path}"
            )

        # Save the DataFrame to a SQLite database if requested
        if self.save_db_path:
            manager = DatabaseManager(self.save_db_path)
            manager.save_to_db(emails_df, table_name="emails_processed")
            self.logger.info(
                f"Email data and dominant topics saved to SQLite database: {self.save_db_path}"
            )

        # Get DataFrame with topics and weights
        self.logger.info(
            f"Formulating DataFrame with ranked topics and weights each word"
        )
        ranked_topics_df = self.topic_distribution(corpus, lda_model, num_words=10)

        # Save the DataFrame to a CSV file if requested
        if self.save_csv_path:
            manager = DatabaseManager(self.save_csv_path)
            manager.save_to_csv(ranked_topics_df)
            self.logger.info(
                f"Ranked topics with words and corresponding weights saved to CSV file: {self.save_csv_path}"
            )

        # Save the DataFrame to a SQLite database if requested
        if self.save_db_path:
            manager = DatabaseManager(self.save_db_path)
            manager.save_to_db(ranked_topics_df, table_name=self.topics_table_name)
            self.logger.info(
                f"Ranked topics with words and corresponding weights saved to SQLite database: {self.save_db_path}"
            )

        # Visualize topics

        # Get the absolute path of the current directory (e.g., src/utils)
        current_dir = os.path.abspath(os.path.dirname(__file__))
        # Navigate up one level to reach the root directory
        root_dir = os.path.abspath(os.path.join(current_dir, "../"))
        self.logger.info(f"Root Directory: {root_dir}")

        path_models = f"{root_dir}/data/models"

        # Visualize with pyLDAvis
        self.visualize_topics(lda_model, save_model_path=f"{path_models}")

        return emails_df, ranked_topics_df

    def visualize_topics(self, lda_model, save_model_path):
        # Create corpus and dictionary
        dictionary, corpus = self.create_corpus()

        # Save model
        if save_model_path:
            lda_model.save(f"{save_model_path}/lda.model")
            self.logger.info(f"LDA model saved to: {save_model_path}/lda.model")

        # Visualize
        vis = gensimvis.prepare(lda_model, corpus, dictionary)

        # Save as HTML
        pyLDAvis.save_html(vis, f"{save_model_path}/lda_visualization.html")
        self.logger.info(
            f"LDA model visualization saved to: {save_model_path}/lda_visualization.html"
        )

        # Display
        self.logger.info(f"Displaying LDA model visualization")
        pyLDAvis.display(vis)

        return vis


if __name__ == "__main__":
    # Get the absolute path of the current directory (e.g., src/utils)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    main_dir = os.path.abspath(os.path.join(current_dir, "../"))
    path_models = f"{main_dir}/data/models"
    print(f"Path for LDA models: {path_models}")

    # Connect to the database (or create it if it doesn't exist)
    connection = sqlite3.connect(f"{main_dir}/data/emails_processed.db")

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Load the dataframe from the SQLite database
    table_name = "emails_processed"
    emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    # Close the connection
    connection.close()

    # Use multiple processors to speed up the process
    topics = TopicModeling(
        emails_df, num_processors=6, save_db_path=f"{main_dir}/data/emails_processed.db"
    )

    emails_df, ranked_topics_df = topics.topic_model(num_passes=10, num_topics=10)

    print(f"Ranked Topics:\n{ranked_topics_df}")
