import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import Graph_preprocessing_functions
import HyperParameters
import Utils as U
import random
import time

def load_data():
    #UserID::Gender::Age::Occupation::Zip-code
    users = pd.read_csv(U.users_data, sep='::', engine='python', 
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'])
    #MovieID::Title::Genres
    movies = pd.read_csv(U.movie_data, sep='::', engine='python', 
                        names=['MovieID', 'Title', 'Genres'], header=None, encoding='ISO-8859-1')
    #UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(U.ratings_data, sep='::', engine='python', 
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'], header=None, encoding='ISO-8859-1')
    
    return users, movies, ratings

def preprocess_data(users, movies, ratings):
    #Dropping irrelevant columns
    users = users.drop(columns=['ZipCode', 'Occupation'])
    ratings = ratings.drop(columns=['Timestamp'])

    #Setting correct data types
    Gender_Map = {'F': 0, 'M': 1}
    users['UserID'] = users['UserID'].astype(int)
    users['Age'] = users['Age'].astype(int)
    users['Gender'] = users['Gender'].map(Gender_Map).astype(int)

    movies['MovieID'] = movies['MovieID'].astype(int)
    # Iterate over the rows of the DataFrame
    for index, movie in movies.iterrows():
        genre_list = movie['Genres'].split('|')  # Use split on the Genres column directly

        # Convert genres to their corresponding indices
        genre_indices = [HyperParameters.GENRES.index(g) for g in genre_list]
        
        # Create one-hot encoded genre list
        one_hot_encoded_genres = [1 if x in genre_indices else 0 for x in range(len(HyperParameters.GENRES))]

        # Assign the one-hot encoded genres back to the DataFrame
        movies.at[index, 'Genres'] = one_hot_encoded_genres

    ratings['UserID'] = ratings['UserID'].astype(int)
    ratings['MovieID'] = ratings['MovieID'].astype(int)
    ratings['Rating'] = ratings['Rating'].astype(float)

    ratings_with_genres = pd.merge(ratings, movies[['MovieID', 'Genres']], on='MovieID', how='left')

    users_with_ratings = pd.merge(users, ratings_with_genres[['UserID', 'MovieID', 'Rating', 'Genres']], on='UserID', how='left')

    return users_with_ratings

def process_data_to_graphs(users):
    # Takes user data and processes it into graphs
    processed_graphs = []
    total = len(users['UserID'].unique())
    start_time = time.time()
    interval_start_time = start_time
    # Iterate over unique UserIDs
    for index, user_id in enumerate(users['UserID'].unique()):
        if index % 25 == 0:
            elapsed = time.time() - interval_start_time
            print(f"Processed {index}/{total} graphs. Interval took {elapsed:.2f} seconds.")
            interval_start_time = time.time()

        # Filter the dataframe for a specific user
        user_ratings = users[users['UserID'] == user_id]
        
        user = {}
        # Extract user's data
        user['user_id'] = user_id
        user['gender'] = list(set(user_ratings['Gender'].values))[0]
        user['age'] = list(set(user_ratings['Age'].values))[0]
        user['movie_ids'] = user_ratings['MovieID'].values
        user['ratings'] = user_ratings['Rating'].values
        user['genres'] = user_ratings['Genres'].values 

        #print(user)

        # Create graph for the user
        graph = Graph_preprocessing_functions.create_graph(user)
        
        # Append graph to processed_graphs
        processed_graphs.append(graph)

    return processed_graphs

'''def process_user(user_data):
    """
    Process a single user's data into a graph.
    """
    user = {
        'user_id': user_data['UserID'],
        'gender': list(set(user_data['Gender']))[0],
        'age': list(set(user_data['Age']))[0],
        'movie_ids': user_data['MovieID'].values,
        'ratings': user_data['Rating'].values,
        'genres': user_data['Genres'].values,
    }
    # Create graph for the user
    return Graph_preprocessing_functions_2.create_graph(user)

def process_data_to_graphs(users, max_threads=1):
    """
    Takes user data and processes it into graphs using multithreading.
    """
    processed_graphs = []
    user_ids = users['UserID'].unique()
    total = len(user_ids)

    print(f"Starting graph processing for {total} users with {max_threads} threads...")

    # Timer for measuring processing intervals
    start_time = time.time()
    interval_start_time = start_time

    # ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_threads) as executor:
        # Prepare user data in advance for parallel processing
        user_data_batches = [
            users[users['UserID'] == user_id]
            for user_id in user_ids
        ]
        
        # Submit tasks
        futures = {executor.submit(process_user, user_data): user_data for user_data in user_data_batches}

        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                graph = future.result()
                processed_graphs.append(graph)
            except Exception as e:
                print(f"Error processing user {futures[future]['UserID'].iloc[0]}: {e}")

            # Log progress and time every 25 graphs
            if i % 25 == 0:
                elapsed = time.time() - interval_start_time
                print(f"Processed {i}/{total} graphs. Interval took {elapsed:.2f} seconds.")
                interval_start_time = time.time()

    # Total time
    total_elapsed = time.time() - start_time
    print(f"Finished processing {total} users into graphs in {total_elapsed:.2f} seconds.")
    return processed_graphs'''

def make_train_test_split(graphs):
    random.shuffle(graphs) #Shuffle the graphs
    training_split = int(len(graphs) * HyperParameters.TRAIN_SPLIT) #80% of the graphs used as training data
    training_graphs = graphs[:training_split] #Training graphs
    testing_graphs = graphs[training_split:] #Testing graphs
    return training_graphs, testing_graphs

def clean_data():
    print('Loading all the data...')
    users, movies, ratings = load_data()

    print('Preprocessing data...')
    users = preprocess_data(users, movies, ratings)

    print('Creating Graphs...')
    graphs = process_data_to_graphs(users)

    print('Creating train/test split...')
    training_graphs, testing_graphs = make_train_test_split(graphs)

    print('Saving Graphs')
    torch.save(training_graphs, U.train_graphs)
    torch.save(testing_graphs, U.test_graphs)
    print("Data cleanup completed successfully.")

if __name__ == "__main__":
    clean_data()
