import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load preprocessed data and similarity matrix
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))   # <-- you need to save this in your ML project

st.title("Movie Recommendation System 🎬")

# Dropdown for movie selection
selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create tags column DataFrame from movies_dict
movies = pd.DataFrame(movies_dict)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)




# Recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Button action
if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    st.write("**Recommended Movies:**")
    for movie in recommendations:
        st.write(movie)

