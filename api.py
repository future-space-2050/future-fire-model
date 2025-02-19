import requests
import logging
import pandas as pd
import threading
from flask import Flask, request, jsonify
from post_recommender.postData import PostData
from post_recommender.CosineSimilarityRecommender import *
from post_recommender.PostEmbeddingManager import *
from who_to_follow.User import User
from who_to_follow.DataProcessor import DataPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from who_to_follow.Modeling import calculate_similarities
from who_to_follow.Recommender import Recommender


import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# File paths
POST_FILE_PATH = "post_recommender/Post DataSet/POST_DATASET.csv"
USERS_FILE_PATH = r"who_to_follow\DataSet\User_profile.csv"
EMBEDDING_FILE_PATH = "post_recommender/Post DataSet/post_embeddings.npy"

backend_api_base_url = "https://future-fire-backend.onrender.com/auth/users/model"

post_manager = PostEmbeddingManager(
    post_file_path=POST_FILE_PATH,
    post_embeddings_file=EMBEDDING_FILE_PATH
)

class APIError(Exception):
    """Custom exception for API errors"""
    pass

def get_api_response(endpoint, timeout=60):
    try:
        response = requests.get(url=endpoint, timeout=timeout)
        logger.info(f"API request to: {response.url} - Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise APIError(f"API communication error: {e}") from e

@app.route('/post_recommendations', methods=['GET'])
def get_post_recommendations():
    try:
        logger.info("==== Received request for post recommendations")
        posts_dataframe = pd.read_csv(POST_FILE_PATH)
        single_user = {"Profession": "Data Science", "Interests": "Dances, Sport, Art", "Interest_Categories": "Hotel", "interacted_posts": [3, 7, 15]}
        recommender = CosineSimilarityRecommender(posts_df=posts_dataframe, user_data=single_user)
        recommended_post_ids = recommender.recommend(11, use_faiss=False)
        
        post_contents = [{"post_id": pid, "content": posts_dataframe.loc[posts_dataframe["post_id"] == pid, "content"].values[0]} for pid in recommended_post_ids]
        return jsonify({"recommended_posts": post_contents})
    except Exception as e:
        logger.error(f"Error getting post recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations', methods=['GET'])
def get_user_recommendations():
    try:
        user_type = request.args.get('type', default=4, type=int)
        user_data_list = get_api_response(f"{backend_api_base_url}?type={user_type}")
        
        user_list = [User(user_data=u).to_dataframe() for u in user_data_list["data"]]
        user_df = pd.concat(user_list)
        data = pd.read_csv(USERS_FILE_PATH)
        data_preprocessor = DataPreprocessor(data, user_df)
        preprocessed_data = data_preprocessor.preprocess()
        cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
        recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
        recommended_ids = recommender.get_recommendations()
        recommended_profiles = data[data['User_ID'].isin(recommended_ids)]
        return jsonify(recommended_profiles.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        return jsonify({"error": str(e)}), 500

def fetch_external_posts():
    try:
        api_url = "https://future-post-service-1.onrender.com/auth/posts/model?type=4"
        data = get_api_response(api_url, timeout=20)
        
        for post_dt in data.get("data", []):
            post_data = PostData({"post_id": post_dt["post_id"], "content": post_dt["content"]}).to_dict()
            post_manager.add_post(new_post=post_data)
    except APIError as ae:
        logger.error(f"API Error: {str(ae)}")
    except Exception as e:
        logger.error(f"General Error: {str(e)}")

def start_flask_app():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    post_fetch_thread = threading.Thread(target=fetch_external_posts)
    flask_thread = threading.Thread(target=start_flask_app)
    
    post_fetch_thread.start()
    flask_thread.start()
    
    post_fetch_thread.join()
    flask_thread.join()
