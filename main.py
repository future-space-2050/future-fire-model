import time
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


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(USERS_FILE_PATH)
posts_dataframe = pd.read_csv(POST_FILE_PATH)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)


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
        user_id = request.args.get('user_id', type=int)
        single_user = data[data['User_ID'] == user_id]
        
        if single_user.empty:
            return jsonify({"error": "User not found"}), 404
        single_user = single_user[["Profession", "Interests", "Interest_Categories",]]
        
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
        user_id = request.args.get('user_id', type=int)

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        logger.info(f"Received recommendation request for user_id: {user_id}")
        if user_id not in data['User_ID'].values:
            return jsonify({"error": "User not found"}), 404

        user_df = data[data['User_ID'] == user_id]
        data_preprocessor = DataPreprocessor(data, user_df)
        preprocessed_data = data_preprocessor.preprocess()

        if preprocessed_data is None:
            return jsonify({"error": "Data preprocessing failed"}), 500

        user_index = data_preprocessor.get_user_index()
        cosine_similarities = calculate_similarities(preprocessed_data, user_index)
        recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
        recommended_indices = recommender.get_recommendations()
        
        print("Recommended Indices: ", recommended_indices)

        if not recommended_indices:
            return jsonify({"message": "No recommendations found"}), 200

        
        recommended_profiles = data.iloc[recommended_indices]
        return jsonify(recommended_profiles.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        return jsonify({"error": str(e)}), 500



def fetch_external_posts():
    while True:
        logger.info("==== Fetching external posts ====")
        try:
            api_url = "https://future-post-service-1.onrender.com/auth/posts/model?type=2"
            data = get_api_response(api_url, timeout=20)
            
            for post_dt in data.get("data", []):
                post_data = PostData({"post_id": post_dt["post_id"], "content": post_dt["content"]}).to_dict()
                post_manager.add_post(new_post=post_data)
        except APIError as ae:
            logger.error(f"API Error: {str(ae)}")
        except Exception as e:
            logger.error(f"General Error: {str(e)}")
        time.sleep(3600)
        

def fetch_and_process_user_data():
    while True:
        logger.info("==== Fetching and processing user data ====")
        api_url = "https://future-fire-backend.onrender.com/auth/users/model?type=4"
        try:
            data = get_api_response(api_url, timeout=20)
            
            if data and "data" in data:
                for user_info in data["data"]:
                    user = User(user_info)
                    user.update_user_data(data)
                    user.save()
                    logger.info(f"Saved user data for user_id: {user.user_id}")
        except APIError as ae:
            logger.error(f"API Error: {str(ae)}")
            
        time.sleep(3600)


def start_flask_app():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    post_fetch_thread = threading.Thread(target=fetch_external_posts)
    user_data_fetch_thread = threading.Thread(target=fetch_and_process_user_data)
    flask_thread = threading.Thread(target=start_flask_app)
    
    post_fetch_thread.start()
    flask_thread.start()
    
    post_fetch_thread.join()
    flask_thread.join()