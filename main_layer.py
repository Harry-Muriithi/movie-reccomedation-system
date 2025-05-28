# Imports
from data_layer import load_and_prepare_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval

# ---------------------------------------
# Demographic Filtering
# ---------------------------------------

def demographic_filtering(df):
    """
    Calculate IMDB weighted rating for each movie and return a sorted DataFrame.
    """
    m = df['vote_count'].quantile(0.9)
    C = df['vote_average'].mean()

    qualified = df.copy().loc[df['vote_count'] >= m].copy()

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False)
    return qualified

# ---------------------------------------
# Popularity-based Top Movies
# ---------------------------------------

def top_popular_movies(df, top_n=10):
    """
    Return top N movies sorted by popularity.
    """
    popular = df.sort_values('popularity', ascending=False).head(top_n)
    return popular

def plot_top_popular_movies(df, top_n=10):
    """
    Plot the top N movies based on popularity.
    """
    top_movies = df[['title', 'popularity']].sort_values(by='popularity', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_movies, y='title', x='popularity', palette='rocket')
    plt.title(f"Top {top_n} Movies by Popularity", fontsize=14)
    plt.xlabel("Popularity")
    plt.ylabel("Movie Title")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()

# ---------------------------------------
# Content-Based Filtering (Plot Overview)
# ---------------------------------------

def prepare_tfidf_matrix(df):
    """
    Prepares TF-IDF matrix and cosine similarity matrix based on 'overview'.
    Returns TF-IDF matrix, cosine similarity matrix, and title-to-index mapping.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return tfidf_matrix, cosine_sim, indices

def get_plot_based_recommendations(title, cosine_sim, indices, df, top_n=10):
    """
    Given a movie title, returns top N similar movies based on plot (overview).
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# ---------------------------------------
# Metadata-Based Filtering (Credits, Genres, Keywords)
# ---------------------------------------

def prepare_metadata_matrix(df):
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            return names[:3] if len(names) > 3 else names
        return []

    df['director'] = df['crew'].apply(get_director)
    for feature in ['cast', 'keywords', 'genres']:
        df[feature] = df[feature].apply(get_list)

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        elif isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

    for feature in ['cast', 'keywords', 'director', 'genres']:
        df[feature] = df[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    df['soup'] = df.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    df.reset_index(drop=True, inplace=True)
    indices = pd.Series(df.index, index=df['title'])

    return cosine_sim, indices

def get_metadata_recommendations(title, cosine_sim, indices, df, top_n=10):
    """
    Return top N similar movies based on metadata features (cast, crew, genres, keywords).
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
