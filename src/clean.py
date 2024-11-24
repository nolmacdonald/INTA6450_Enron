import numpy as np
import os
import pandas as pd
import sqlite3
import email

from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if __name__ == "__main__":
    # Get the absolute path of the current directory (e.g., src/utils)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Current directory: {current_dir}")
    # Navigate up one level to reach the root directory
    root_dir = os.path.abspath(os.path.join(current_dir, "../"))
    print(f"Root directory: {root_dir}")

    # INTA6450_Enron/data/emails.db
    path_db = f"{root_dir}/data/emails.db"

    # Table name for DataFrame saved in the database
    table_name = "emails"

    # Connect to the database (or create it if it doesn't exist)
    connection = sqlite3.connect(path_db)

    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()

    # Load the dataframe from the SQLite database
    emails_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    # Close the connection
    connection.close()

    # Show email data
    print(f"Number of emails: {len(emails_df)}")
    print(emails_df.head())

    # Get the first email text
    first_email_text = emails_df.iloc[0]["text"]
