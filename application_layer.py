import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import requests
from itertools import zip_longest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_layer import load_and_prepare_data
from model_layer import load_ratings_data, CollaborativeFiltering

# --- Load & Prepare Data & Model ---
df = load_and_prepare_data()   # includes df['poster_path'] and df['id']
ratings_df = load_ratings_data()
cf = CollaborativeFiltering(ratings_df)

# --- Streamlit Config ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.success("✅ Data & CF model loaded!")

# --- Constants ---
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_IMAGES = "https://api.themoviedb.org/3/movie/{}/images"
HEADERS = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlZDgzOTQyODZhY2FmN2Y1ODg0ZGMwNGYxMzM5M2MwMyIsIm5iZiI6MTc0ODIwODkyNy4zNTM5OTk5LCJzdWIiOiI2ODMzOGQxZjExZGNlZTg3MzU4MzdiY2IiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.NaDGWTV0eWKe7mG1qJs38RxYmgaFEsuRjx4yEipcU_k"
}
PLACEHOLDER = "https://via.placeholder.com/342x513?text=No+Image"
DEFAULT_COLS = 5

# --- Helper: Fetch poster by Movie ID ---
def fetch_poster(movie_id):
    """Fetch poster file path from TMDB API given a movie ID."""
    try:
        url = TMDB_API_IMAGES.format(movie_id)
        resp = requests.get(url, headers=HEADERS, timeout=5)
        data = resp.json()
        posters = data.get('posters', [])
        if posters:
            # pick the first poster
            file_path = posters[0].get('file_path')
            return TMDB_IMAGE_BASE + file_path
    except Exception:
        pass
    return PLACEHOLDER

# --- Helper: Grid Display with dynamic fetch ---
def show_movies_grid(rows, cols=DEFAULT_COLS):
    """Display movie posters and titles in a grid layout, fetching by ID."""
    for group in zip_longest(*[iter(rows)] * cols, fillvalue=None):
        cols_layout = st.columns(cols)
        for cell, layout in zip(group, cols_layout):
            if cell is None:
                continue
            _, row = cell
            movie_id = row.get('id')
            img_url = fetch_poster(movie_id) if movie_id else PLACEHOLDER
            layout.image(img_url, width=180)
            layout.markdown(f"<p style='text-align:center; font-weight:bold;'>{row['title']}</p>", unsafe_allow_html=True)

st.markdown("---")

# 1) Popularity Chart in Expander
with st.expander("📊 Top Movies by Popularity", expanded=True):
    def plot_popularity(df, n=10):
        top = df.nlargest(n, 'popularity')[['title','popularity']]
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=top, y='title', x='popularity', palette='rocket', ax=ax)
        ax.set_title(f"Top {n} Movies by Popularity")
        st.pyplot(fig)
    plot_popularity(df)

st.markdown("---")

# 2) Demographic Sampling
with st.expander("🎯 Demographic-Based Recommendations", expanded=False):
    def demographic(df):
        m = df['vote_count'].quantile(0.9)
        C = df['vote_average'].mean()
        qual = df[df['vote_count'] >= m].copy()
        qual['score'] = qual.apply(
            lambda x: (x['vote_count']/(x['vote_count']+m)*x['vote_average']) + 
                      (m/(m+x['vote_count'])*C), axis=1)
        return qual.sort_values('score', ascending=False)

    demo_df = demographic(df)
    if st.button("Show 5 Demographic Recs"):
        show_movies_grid(demo_df.sample(5).iterrows())

st.markdown("---")

# 3) Genre-Based Recommendations
with st.expander("🔍 Genre-Based Recommendations", expanded=False):
    movie_choice = st.selectbox("Choose a Movie:", df['title'], key='genre_select')
    if st.button("Get Genre Recs"):
        sel = df[df['title'] == movie_choice]
        if not sel.empty:
            genres = [g['name'] for g in ast.literal_eval(sel.iloc[0]['genres'])]
            mask = df['genres'].apply(
                lambda s: any(name in [x['name'] for x in ast.literal_eval(s)] for name in genres)
            )
            recs = demographic(df[mask & (df['title'] != movie_choice)]).head(5)
            show_movies_grid(recs.iterrows())

st.markdown("---")

# 4) Plot-Based Recommendations
with st.expander("🔍 Plot-Based Recommendations", expanded=False):
    tfidf = TfidfVectorizer(stop_words='english')
    mat = tfidf.fit_transform(df['overview'].fillna(''))
    sim = cosine_similarity(mat, mat)
    idx = pd.Series(df.index, index=df['title']).drop_duplicates()

    plot_movie = st.selectbox("Choose a Movie:", df['title'], key='plot_select')
    if st.button("Get Plot Recs"):
        i = idx[plot_movie]
        vals = sorted(enumerate(sim[i]), key=lambda x: x[1], reverse=True)[1:6]
        recs = df.iloc[[v for v,_ in vals]]
        show_movies_grid(recs.iterrows())

st.markdown("---")

# 5) Metadata-Based Recommendations
with st.expander("🔍 Metadata-Based Recommendations", expanded=False):
    # Preprocess metadata once
    for c in ['cast','crew','keywords','genres']:
        df[c] = df[c].apply(ast.literal_eval)
    df['director'] = df['crew'].apply(lambda x: next((i['name'] for i in x if i['job'] == 'Director'), ''))
    for c in ['cast','keywords','genres']:
        df[c] = df[c].apply(lambda lst: [i['name'] for i in lst][:3])
    df['soup'] = df.apply(lambda x: ' '.join(x['keywords'] + x['cast'] + [x['director']] + x['genres']), axis=1)
    mat2 = CountVectorizer(stop_words='english').fit_transform(df['soup'])
    sim2 = cosine_similarity(mat2, mat2)
    idx2 = pd.Series(df.index, index=df['title']).drop_duplicates()

    meta_movie = st.selectbox("Choose a Movie:", df['title'], key='meta_select')
    if st.button("Get Meta Recs"):
        i = idx2[meta_movie]
        vals = sorted(enumerate(sim2[i]), key=lambda x: x[1], reverse=True)[1:6]
        recs = df.iloc[[v for v,_ in vals]]
        show_movies_grid(recs.iterrows())

st.markdown("---")

# 6) Collaborative Filtering Prediction
with st.expander("🤝 CF Rating Prediction", expanded=False):
    uid = st.number_input("User ID", min_value=1, step=1, key='uid')
    mid = st.number_input("Movie ID", min_value=1, step=1, key='mid')
    if st.button("Predict Rating"):
        est = cf.predict_rating(uid, mid)
        st.write(f"**Predicted rating:** {est:.2f}")
