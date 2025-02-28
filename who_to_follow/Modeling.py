from sklearn.metrics.pairwise import cosine_similarity
from who_to_follow.DataProcessor import DataPreprocessor
from who_to_follow.error_message import *
from who_to_follow.User import User
from who_to_follow.Recommender import Recommender


import pandas as pd


def calculate_similarities(preprocessed_data, user_index):
    """Calculate cosine similarities between the target user and all other users."""
    data_matrix = preprocessed_data.values
    target_vector = data_matrix[user_index].reshape(1, -1)
    print("Calculating cosine similarities")
    return cosine_similarity(target_vector, data_matrix).flatten()


def load_data(filename):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError("The specified file does not exist.")
    except pd.errors.ParserError:
        raise DataParserError("Error while parsing the CSV file.")
 


def runner(user_data, USERS_FILE_PATH):
    data = load_data(USERS_FILE_PATH)
    data_preprocessor = DataPreprocessor(data, user_data)
    preprocessed_data = data_preprocessor.preprocess()
    cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
    recommender = Recommender(preprocessed_data, cosine_similarities, top=20)
    recommendations = recommender.get_recommendations()
    data_frame_recommendations = data.iloc[recommendations]["userID"]
    return data_frame_json(data_frame_recommendations)
    
    
    
# pandas DataFrame into json format
def data_frame_json(data_frame):
    """Convert a pandas DataFrame to JSON format."""
    return data_frame.to_json(orient='records', date_format='iso')
