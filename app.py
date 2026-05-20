import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
print(os.getcwd())

# Select required columns
movies = movies[['title', 'overview']]

# Remove null values
movies.dropna(inplace=True)

# Convert text to lowercase
movies['overview'] = movies['overview'].apply(lambda x: x.lower())

# Convert text into vectors
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(movies['overview']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")

selected_movie = st.selectbox(
    "Select Movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    for movie in recommendations:
        st.write(movie)