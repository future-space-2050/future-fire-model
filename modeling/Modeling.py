from sklearn.metrics.pairwise import cosine_similarity
from DataProcessor import DataPreprocessor
from Recommender import Recommender
import pandas as pd


FILE_PATH = r"Modeling\DataSet\User_profile.csv"
data = pd.read_csv(FILE_PATH)
data = data.drop(columns=["max_prob"])


user = data.iloc[12:13]

def calculate_similarities(preprocessed_data, user_index):
    """Calculate cosine similarities between the target user and all other users."""
    data_matrix = preprocessed_data.values
    target_vector = data_matrix[user_index].reshape(1, -1)
    return cosine_similarity(target_vector, data_matrix).flatten()


data_preprocessor = DataPreprocessor(data, user)
preprocessed_data = data_preprocessor.preprocess()
cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
recommendations = recommender.get_recommendations()
print(recommendations)

print((data.iloc[recommendations]))

print(user)