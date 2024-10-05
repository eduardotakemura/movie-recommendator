import streamlit as st
import numpy as np
import pandas as pd
import random

## --------------- PREDICTIVE METHOD --------------- ##
# Predictive method #
def predict(movie_ids, ratings, movie_embedding_layer):
    # Get movie embeddings for the rated movies
    movie_embeddings = movie_embedding_layer[movie_ids]

    # Compute weighted movie embeddings and average them to create the user embedding
    new_user_embedding = np.mean(movie_embeddings * np.array(ratings).reshape(-1, 1), axis=0)

    # Predict ratings by taking the dot product between user and all movies
    predicted_ratings = np.dot(movie_embedding_layer, new_user_embedding)

    # Get top recommended movies
    top_movie_ids = np.argsort(-predicted_ratings)[:10]  # Top 10 movie ids
    return top_movie_ids

## --------------- LOADING DATA --------------- ##
@st.cache_resource
def load_data():
    """Load movies embeddings and movies dataframe from training, and store it on cache."""
    movie_embeddings = np.load('data/movie_embedding_gmf.npy')
    movies_df = pd.read_csv('data/movies.csv')
    return movie_embeddings, movies_df

movie_embedding_gmf, df_movies = load_data()
movie_list = df_movies['title'].tolist()

# ------ SESSION ------ #
# Initialize movies, if there's no data stored in session #
if 'user_ratings' not in st.session_state:
    num_movies_to_show = 10
    selected_movies = random.sample(movie_list, num_movies_to_show) if len(
        movie_list) > num_movies_to_show else movie_list

    st.session_state['user_ratings'] = {movie: 5.0 for movie in selected_movies}

# Else, load previous data #
selected_movies = list(st.session_state['user_ratings'].keys())

## --------------- STREAMLIT APP --------------- ##
# ------ HEADER ------ #
st.title("Movie Recommendator")
st.write("Hi there! I'm a Neural Collaborative Filtering (NCF) model, and in my current version, "
         "I've been trained on over 100,000 user ratings spanning a collection of more than 10,000 movies.")
st.write("My architecture leverage deep learning neural networks to provide personalized movie recommendations. "
         "My training incorporates not just user ratings but also movie metadata such as year of release and genres, "
         "making my predictions more nuanced.")
st.divider()
st.subheader("Try for yourself! Rate the following movies, and I'll suggest others you might like.")

# ------ MOVIES SLIDERS ------ #
for movie in selected_movies:
    year = int(df_movies.loc[df_movies['title'] == movie, 'year'].values[0])
    st.session_state['user_ratings'][movie] = st.slider(f"**{movie} ({year})**", 0.0, 10.0, st.session_state['user_ratings'][movie])

# ------ BUTTON ------ #
if st.button('Get Recommendations'):
    # Get rated movie ids and corresponding ratings #
    rated_movies = [i for i, movie in enumerate(movie_list) if st.session_state['user_ratings'].get(movie, 0) > 0][:10]
    user_ratings_list = [rating/10 for rating in st.session_state['user_ratings'].values() if rating > 0]

    # Get top 10 predictions #
    top_movie_ids = predict(
        movie_ids=rated_movies,
        ratings=user_ratings_list,
        movie_embedding_layer=movie_embedding_gmf
    )

    # Display recommendations #
    st.divider()
    st.write("### Recommended Movies")
    for movie_id in top_movie_ids:
        with st.container():
            title = df_movies.iloc[movie_id]['title']
            year = int(df_movies.iloc[movie_id]['year'])
            st.markdown(f"**:film_frames: {title} ({year})**")
