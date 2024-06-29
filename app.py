import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def first4(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 4:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


tfidf_vectorizer_path = 'tfidf_vectorizer.pkl'
tfidf_matrix_path = 'tfidf_matrix.pkl'
movies_path = 'movies.pkl'

if os.path.exists(tfidf_vectorizer_path) and os.path.exists(tfidf_matrix_path) and os.path.exists(movies_path):
    # Load the TF-IDF vectorizer and matrix from pickle files
    with open(tfidf_vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with open(tfidf_matrix_path, 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open(movies_path, 'rb') as file:
        movies = pickle.load(file)
else:
    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the TF-IDF model and transform the 'tags' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['tags'])

    # Pickle the TF-IDF vectorizer and matrix
    with open(tfidf_vectorizer_path, 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    with open(tfidf_matrix_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)
    with open(movies_path, 'wb') as file:
        pickle.dump(movies, file)

def recommend(movie_id, n=5):
    if movie_id not in movies['movie_id'].values:
        return "Movie not found"

    # Find the movie in the dataset
    movie_index = movies[movies['movie_id'] == movie_id].index[0]

    # Get the TF-IDF vector for the given movie
    movie_vector = tfidf_matrix[movie_index]

    # Calculate similarity scores
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()

    # Get the indices of the top n similar movies
    similar_indices = similarities.argsort()[-n-1:-1][::-1]

    # Return a list of objects with movie id and title
    recommendations = [{"id": int(movies.iloc[idx]['movie_id']), "title": movies.iloc[idx]['title']} for idx in similar_indices]

    return recommendations



def add_new_movie(new_movie):
    global tfidf_vectorizer, tfidf_matrix, movies

    # Process the new movie data
    new_movie['genres'] = ast.literal_eval(new_movie['genres'])
    new_movie['keywords'] = ast.literal_eval(new_movie['keywords'])
    new_movie['cast'] = first4(new_movie['cast'])
    new_movie['crew'] = fetch_director(new_movie['crew'])

    new_movie['overview'] = new_movie['overview'].split()
    new_movie['genres'] = [i.replace(" ", "") for i in new_movie['genres']]
    new_movie['keywords'] = [i.replace(" ", "") for i in new_movie['keywords']]
    new_movie['cast'] = [i.replace(" ", "") for i in new_movie['cast']]
    new_movie['crew'] = [i.replace(" ", "") for i in new_movie['crew']]

    # Concatenate all features into a single 'tags' column
    new_movie['tags'] = new_movie['overview'] + new_movie['genres'] + new_movie['keywords'] + new_movie['cast'] + new_movie['crew']
    new_movie['tags'] = " ".join(new_movie['tags']).lower()

    # Add the new movie to the movies DataFrame
    new_movie_df = pd.DataFrame([new_movie])
    movies = pd.concat([movies, new_movie_df], ignore_index=True)

    # Update the TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['tags'])

    # Re-pickle the updated TF-IDF vectorizer, matrix, and movies
    with open(tfidf_vectorizer_path, 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    with open(tfidf_matrix_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)
    with open(movies_path, 'wb') as file:
        pickle.dump(movies, file)






from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Example function to get movie recommendations (to be replaced with your model)
def get_movie_recommendations(movie_id,quantity):
    # Dummy data for illustration
    recommendations = recommend(movie_id,quantity)
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommended():
    data = request.json
    tmdb_id = data.get('tmdb_id')
    quantity = data.get('quantity')

    if not quantity:
        quantity = 5
        
    if quantity > 200:
        return jsonify({"error": "Maximum quantity should be less than 200"}), 400

    if not tmdb_id :
        return jsonify({"error": "tmdb_id is required"}), 400
    else:
        recommendations = get_movie_recommendations(tmdb_id , quantity)

    return jsonify({"recommendations": recommendations})



@app.route('/addData', methods=['POST'])
def addData():
    try:
        # Parse JSON data from the request
        movies = request.json

        if not isinstance(movies, list):
            return jsonify({'error': 'Data should be a list of movie objects'}), 400

        for new_movie in movies:
            required_keys = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
            for key in required_keys:
                if key not in new_movie:
                    return jsonify({'error': f'Missing key: {key} in movie object'}), 400

            # No need to convert strings to lists/dictionaries here

            # Add the movie using the provided function
            add_new_movie(new_movie)

        return jsonify({'message': 'Movies added successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


