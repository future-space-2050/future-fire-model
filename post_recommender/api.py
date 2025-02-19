import requests
import logging
import pandas as pd
from flask import Flask, request, jsonify
from postData import PostData
from CosineSimilarityRecommender import *
from PostEmbeddingManager import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

backend_api_base_url = r"API"
app = Flask(__name__)

post_manager = PostEmbeddingManager(
    post_file_path=POST_FILE_PATH,
    post_embeddings_file=EMBEDDING_FILE_PATH
)

class APIError(Exception):
    """Custom exception for API errors"""
    pass

def get_api_response(endpoint, params=None, headers=None, timeout=60):
    try:
        if not isinstance(endpoint, str) or not endpoint.startswith(('http://', 'https://')):
            raise ValueError("Invalid endpoint URL")

        response = requests.get(url=endpoint, timeout=timeout)
        logger.info(f"API request to: {response.url} - Status: {response.status_code}")
        response.raise_for_status()

        if 'application/json' not in response.headers.get('Content-Type', ''):
            raise ValueError("Unexpected content type in response")

        return response.json()

    except requests.Timeout as te:
        logger.error(f"API request timed out: {te}")
        raise APIError("Request timed out") from te
    except requests.RequestException as re:
        logger.error(f"API request failed: {str(re)}")
        raise APIError(f"API communication error: {str(re)}") from re
    except requests.JSONDecodeError:
        logger.error("Failed to parse JSON response")
        raise APIError("Invalid JSON response")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise APIError("Unknown error occurred") from e

@app.route('/post_recommendations', methods=['GET'])
def get_recommendation():
    try:
        logger.info("==== Received request for recommendations")

        # Load posts dataset
        posts_dataframe = pd.read_csv(POST_FILE_PATH)

        # Example user profile
        single_user = {
            "Profession": "Data Science",
            "Interests": "Dances, Sport, Art",
            "Interest_Categories": "Hotel",
            "interacted_posts": [3, 7, 15]
        }

        recommender = CosineSimilarityRecommender(posts_df=posts_dataframe, user_data=single_user)
        recommended_post_ids = recommender.recommend(11, use_faiss=False)

        post_contents = []
        for post_id in recommended_post_ids:
            content = posts_dataframe.loc[posts_dataframe["post_id"] == post_id, "content"].values
            if len(content) > 0:
                post_contents.append({"post_id": post_id, "content": content[0]})

        return jsonify({"recommended_posts": post_contents})

    except Exception as e:
        error_message = f"Error getting recommendations: {e}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    try:
        logger.info("==== Starting application ====")

        posts = pd.read_csv(POST_FILE_PATH)
        users = pd.read_csv(USERS_FILE_PATH)

        

        logger.info("==== Fetching posts from external API ====")
        api_url = "https://future-post-service-1.onrender.com/auth/posts/model?type=4"
        data = get_api_response(api_url, timeout=20)
        logger.info("==== Fetching  ====")
        post_manager = PostEmbeddingManager(
            post_file_path=POST_FILE_PATH,
            post_embeddings_file=EMBEDDING_FILE_PATH
        )
        for post_dt in data["data"]:
            post_data_format = {
                "post_id": post_dt["post_id"],
                "content": post_dt["content"],
            }
            post_data = PostData(post_data_format).to_dict()
            print(
                post_data
            )
            post_manager.add_post(
                new_post=post_data
            )
            
    except APIError as ae:
        logger.error(f"API Error: {str(ae)}")
    except Exception as e:
        logger.error(f"General Error: {str(e)}")
