import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed movie data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Compute similarity at runtime
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Streamlit UI
st.title("Movie Recommendation System 🎬")

# Dropdown for movie selection
selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

# Recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]
    return recommended_movies

# Button action
if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    st.write("**Recommended Movies:**")
    for movie in recommendations:
        st.write(movie)
