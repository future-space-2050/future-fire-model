import os
import pandas as pd
from who_to_follow.Recommender import Recommender
from who_to_follow.Modeling import calculate_similarities, load_data 
from who_to_follow.DataProcessor import DataPreprocessor


from collections import defaultdict


DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DataSet" 
)

def get_dataset_filenames():
    """Returns a list of all CSV files in the dataset directory."""
    if not os.path.isdir(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        return []

    return [
        file for file in os.listdir(DATASET_DIR)
        if os.path.isfile(os.path.join(DATASET_DIR, file)) and 
        file.startswith("user_profile_data_") and 
        file.endswith(".csv")
    ]

def find_user_profile_data(user_id):
    """Searches for user profile data across multiple CSV files."""
    dataset_filenames = get_dataset_filenames()

    for filename in dataset_filenames:
        file_path = os.path.join(DATASET_DIR, filename)
        data = load_data(file_path)  
        user_data = data.loc[data["User_ID"] == user_id]

        if not user_data.empty:
            return user_data

    print(f"User profile data for user ID {user_id} not found.")
    return None


def calculate_cosine_simmilarity(user_id):
    user_data = find_user_profile_data(user_id)
    if user_data is None:
        return None
    
    top_simmilar_users = []
    top_20_simmilar_users_per_file = defaultdict(list)
    
    for file in get_dataset_filenames():
        file_path = os.path.join(DATASET_DIR, file)
        other_user_data = load_data(file_path)
        
        # check if user_id is already
        if user_id not in other_user_data["User_ID"].values:
            # Add user data to dataset 
            other_user_data = pd.concat([other_user_data, user_data], ignore_index=True)

        
        preprocessor = DataPreprocessor(
            other_user_data,
            user=user_data
        )
        
        processed_data = preprocessor.preprocess()
        user_index = preprocessor.get_user_index()
        cosine_similarities = calculate_similarities(processed_data, user_index)
        recommender = Recommender(processed_data, cosine_similarities, top=1)
        recommended_indices = recommender.get_recommendations()
        top_20_simmilar_users_per_file[file] = recommended_indices
        
        # Add top 20 similar users to top_simmilar_users
        top_simmilar_users = get_recommended_indices(top_20_simmilar_users_per_file)
        
    return top_simmilar_users

def get_recommended_indices(top_20_simmilar_users_per_file):
    
    top_20_simmilar_users_per_file = [
        (degree, file, index) for file in top_20_simmilar_users_per_file for index, degree in top_20_simmilar_users_per_file[file]
    ]
    top_20_simmilar_users_per_file.sort(reverse=True)
    top_20_simmilar_users_per_file = top_20_simmilar_users_per_file[:20]
    
    top_20_simmilar_users = []
    for _, file, index in top_20_simmilar_users_per_file:
        data = load_data(os.path.join(DATASET_DIR, file))
        user_data = data.iloc[index]
        top_20_simmilar_users.append(user_data.to_dict())
        
    return top_20_simmilar_users