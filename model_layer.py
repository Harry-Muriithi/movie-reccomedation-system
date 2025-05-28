# model_layer.py

import kagglehub
import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

def load_ratings_data():
    """
    Downloads and loads the ratings dataset from the MovieLens Kaggle dataset.

    Returns:
        pd.DataFrame: Ratings DataFrame with columns ['userId', 'movieId', 'rating', 'timestamp']
    """
    try:
        dataset_path = kagglehub.dataset_download("grouplens/movielens-latest-small")
        ratings_csv = os.path.join(dataset_path, "ratings.csv")
        if not os.path.exists(ratings_csv):
            raise FileNotFoundError(f"'ratings.csv' not found in {dataset_path}")

        df = pd.read_csv(ratings_csv)
        expected = {'userId','movieId','rating','timestamp'}
        if not expected.issubset(df.columns):
            raise ValueError(f"Missing columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load ratings data: {e}")

class CollaborativeFiltering:
    def __init__(self, ratings_df: pd.DataFrame):
        """
        Initialize with a preloaded ratings DataFrame.
        """
        self.ratings_df = ratings_df
        self.model = SVD()
        # build Surprise dataset immediately
        reader = Reader(rating_scale=(0.5, 5.0))
        self.data = Dataset.load_from_df(
            ratings_df[['userId','movieId','rating']],
            reader
        )
        self.trainset = None

    def evaluate_model(self):
        """
        Perform 5-fold cross-validation, returning RMSE and MAE.
        """
        return cross_validate(self.model, self.data, measures=['RMSE','MAE'], cv=5, verbose=True)

    def train_model(self):
        """
        Train the SVD model on the full dataset.
        """
        self.trainset = self.data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id, movie_id, actual_rating=None):
        """
        Predict a user's rating for a movie.
        """
        return self.model.predict(uid=user_id, iid=movie_id, r_ui=actual_rating)

    def get_user_ratings(self, user_id):
        """
        Return all ratings from one user.
        """
        return self.ratings_df[self.ratings_df['userId']==user_id]
