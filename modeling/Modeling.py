from sklearn.metrics.pairwise import cosine_similarity
from DataProcessor import DataPreprocessor
from ErrorMessage import *
# Json to dataframe
from User import User
from Recommender import Recommender

import pandas as pd


def calculate_similarities(preprocessed_data, user_index):
    """Calculate cosine similarities between the target user and all other users."""
    print(preprocessed_data)
    data_matrix = preprocessed_data.values
    target_vector = data_matrix[user_index].reshape(1, -1)
    return cosine_similarity(target_vector, data_matrix).flatten()
 
    
# user_data = {
#     "User_ID": 10,
#     "Name": "Kiilolee Daawwitii",
#     "City": "Muger",
#     "Birth_Date": "2007-06-21",
#     "Interests": ["Songwriting", "Filmmaking", "Sculpting", "Art", "Coaching"],
#     "Gender": "Male",
#     "Interest_Categories": ["Animals_Nature", "Arts_Creativity", "Science_Technology"],
#     "Profession": "Digital Content Creator",
#     "Latitude": 8.582099,
#     "Longitude": 38.885129,
#     "Birth_Year": 2007,
#     "Age": 17
# }


# data = pd.read_csv(r"Modeling\Dataset\user_profile.csv")
# user = User(user_data=user_data)

# user = user.to_data_frame()
# print(user)

# data_preprocessor = DataPreprocessor(data, user)
# preprocessed_data = data_preprocessor.preprocess()
# cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
# recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
# recommendations = recommender.get_recommendations()
# print(recommendations)


# print(data.iloc[recommendations])