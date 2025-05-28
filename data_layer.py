import pandas as pd
import kagglehub
import os

def load_and_prepare_data():
    """
    Download and load the TMDB movie datasets from KaggleHub,
    merge movies and credits datasets, clean missing/duplicate data,
    and return a combined DataFrame.
    """
    path = kagglehub.dataset_download("chaitanyasood1/tmdb-5000-movies")

    movies_path = os.path.join(path, "tmdb_5000_movies.csv")
    credits_path = os.path.join(path, "tmdb_5000_credits.csv")

    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    merged_df = pd.merge(movies_df, credits_df, on='title', how='inner')
    merged_df.dropna(inplace=True)
    merged_df.drop_duplicates(inplace=True)

    return merged_df
