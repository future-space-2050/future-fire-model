import time
from xml.dom import ValidationErr
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
from who_to_follow.Modeling import calculate_similarities, runner
from who_to_follow.Recommender import Recommender
from who_to_follow.simmilarity_calculator import calculate_cosine_simmilarity


import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(USERS_FILE_PATH)
posts_dataframe = pd.read_csv(POST_FILE_PATH)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)


POST_FILE_PATH = r"post_recommender/Post Dataset/POST_DATASET.csv"
USERS_FILE_PATH = r"who_to_follow/DataSet/User_profile.csv"
EMBEDDING_FILE_PATH = r"post_recommender/Post Dataset/POST_EMBEDDINGS.npy"

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
        single_user = data[data['userID'] == user_id]
        
        if single_user.empty:
            return jsonify({"error": "User not found"}), 404
        single_user = single_user[["occupation", "interests", "interestCategories",]]
        recommender = CosineSimilarityRecommender(posts_df=posts_dataframe, user_data=single_user)
        recommended_post_ids = recommender.recommend(11, use_faiss=False)
        post_contents = [{"post_id": pid, "content": posts_dataframe.loc[posts_dataframe["post_id"] == pid, "content"].values[0]} for pid in recommended_post_ids]
        return jsonify({"recommended_posts": post_contents})
    except Exception as e:
        logger.error(f"Error getting post recommendations: {e}")
        return jsonify({"error post recommendation": str(e)}), 500




@app.route('/recommendations', methods=['GET'])
def get_user_recommendations():
    try:
        user_id = request.args.get('user_id', type=int)

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        logger.info(f"Received recommendation request for user_id: {user_id}")
        
        # Ensure `data` is available and contains 'userID'
        if user_id not in data['userID'].values:
            return jsonify({"error": "User not found"}), 404
        
        user_df = data[ data['userID'] == user_id]
        
        recommended_profiles = runner(user_df, USERS_FILE_PATH)
        if len(recommended_profiles) == 0:
            return jsonify({"message": "No recommendations found"}), 200

        # Return the DataFrame as JSON
        return recommended_profiles
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        return jsonify({"error who to follow": str(e)}), 500

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
        api_url = "https://future-fire-backend.onrender.com/auth/users/model?type=2"
        try:
            data = get_api_response(api_url, timeout=20)
            
            if data and "data" in data:
                for user_info in data["data"]:
                    user = User(user_info)
                    user.save()
                    logger.info(f"Saved user data for user_id: {user.user_id}")
        except APIError as ae:
            logger.error(f"API Error: {str(ae)}")
        
        data = pd.read_csv(USERS_FILE_PATH)
        time.sleep(3600)

     
@app.route('/get_users', methods=['GET'])
def get_users_data():
    """Endpoint to fetch user data from CSV or external API"""
    try:
        logger.info(f"Request received from {request.remote_addr} for user data")
        
        # Validate and parse input parameter
        try:
            data_type = int(request.args.get('type', -1))
        except ValueError:
            logger.warning(f"Invalid type parameter format: {request.args.get('type')}")
            return jsonify({"error": "Type must be an integer"}), 400

        # Handle CSV data request
        if data_type == -1:
            try:
                df = pd.read_csv("DataSets/User_profile.csv", usecols=[
                    'userID', 'fullName', 'occupation', 'interests', 'location'
                ])
                logger.info(f"Returning {len(df)} users from CSV")
                return jsonify({"users": df.where(pd.notnull(df), None).to_dict(orient='records')}), 200
            except FileNotFoundError:
                logger.error("User profile CSV file not found")
                return jsonify({"error": "Data source unavailable"}), 503

        # Handle API-based requests
        elif 1 <= data_type <= 4:
            api_url = f"https://future-fire-backend.onrender.com/auth/users/model?type={data_type}"
            
            try:
                # Make API call with timeout and retry logic
                response = requests.get(
                    api_url,
                    timeout=(3.05, 20),  # Connect timeout 3s, read timeout 20s
                    headers={'Accept': 'application/json'}
                )
                response.raise_for_status()

                # Parse and validate response
                data = response.json()
                if not isinstance(data.get('data'), list):
                    logger.error("Invalid API response format")
                    return jsonify({"error": "Unexpected data format"}), 502

                # Process users
                processed_users = pd.DataFrame()
                for user_info in data["data"]:
                    try:
                        user = User(user_data=user_info)
                        processed_user = user.to_dataframe()
                        processed_users = pd.concat(
                            [processed_users, processed_user],
                            axis=0,
                            ignore_index=True
                        )
                    except ValidationErr as ve:
                        logger.warning(f"Skipping invalid user data: {str(ve)}")
                        continue
                processed_users = processed_users[['userID', 'fullName', 'occupation', 'interests', 'location']]
                processed_users = processed_users.where(pd.notnull(processed_users), None).to_dict(orient='records')

                logger.info(f"Returning {len(processed_users)} processed users from API")
                return jsonify({
                    "users": processed_users,
                    "metadata": {
                        "source": api_url,
                        "count": len(processed_users)
                    }
                }), 200

            except requests.exceptions.Timeout:
                logger.error("API request timed out")
                return jsonify({"error": "External service timeout"}), 504
            except requests.exceptions.JSONDecodeError:
                logger.error("Failed to decode API response")
                return jsonify({"error": "Invalid API response"}), 502
            except requests.exceptions.RequestException as e:
                logger.error(f"API connection error: {str(e)}")
                return jsonify({"error": "External service unavailable"}), 503

        # Handle invalid type parameters
        else:
            logger.warning(f"Invalid type parameter: {data_type}")
            return jsonify({
                "error": "Invalid request parameter",
                "valid_values": [-1, 1, 2, 3, 4]
            }), 400

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500    

def start_flask_app():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    
    post_fetch_thread = threading.Thread(target=fetch_external_posts)
    user_data_fetch_thread = threading.Thread(target=fetch_and_process_user_data)
    flask_thread = threading.Thread(target=start_flask_app)
    
    post_fetch_thread.start()
    flask_thread.start()
    user_data_fetch_thread.start()

    
    post_fetch_thread.join()
    flask_thread.join()
    user_data_fetch_thread.join()