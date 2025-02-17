import requests
from flask import Flask, request, jsonify
from postData import PostData
from CosineSimilarityRecommender import *
from PostEmbeddingManager import *
from flask import COE

backend_api_base_url = r"https://future-post-service-1.onrender.com/auth/posts/model"
app = Flask(__name__)

post_manager = PostEmbeddingManager(
    post_file_path=POST_FILE_PATH,
    post_embeddings_file=EMBEDDING_FILE_PATH
)

recommender = CosineSimilarityRecommender(post_manager)

# ask the post manager to create a new post

import requests
from requests.exceptions import RequestException, Timeout, JSONDecodeError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_response(endpoint, params=None, headers=None, timeout=60):
    """
    Safely retrieve data from an API endpoint
    
    Args:
        endpoint (str): API URL
        params (dict): Query parameters
        timeout (int): Timeout in seconds
    
    Returns:
        dict: API response data or None
    
    Raises:
        APIError: For custom error handling
    """
    try:
        if not isinstance(endpoint, str) or not endpoint.startswith(('http://', 'https://')):
            raise ValueError("Invalid endpoint URL")

        response = requests.get(
            url=endpoint,
            timeout=timeout
        )

        logger.info(f"API request to: {response.url} - Status: {response.status_code}")

        response.raise_for_status()

        if 'application/json' not in response.headers.get('Content-Type', ''):
            raise ValueError("Unexpected content type in response")

        try:
            return response.json()
        except JSONDecodeError:
            logger.error("Failed to parse JSON response")
            raise ValueError("Invalid JSON response")

    except Timeout as te:
        logger.error(f"API request timed out: {te}")
        raise APIError("Request timed out") from te
    except RequestException as re:
        logger.error(f"API request failed: {str(re)}")
        raise APIError(f"API communication error: {str(re)}") from re
    except (ValueError, KeyError) as ve:
        logger.error(f"Data validation error: {str(ve)}")
        raise APIError("Data processing error") from ve
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise APIError("Unknown error occurred") from e

class APIError(Exception):
    """Custom exception for API errors"""
    pass

# Example usage:
if __name__ == "__main__":
    try:
        n = 3
        api_url = f"https://future-post-service-1.onrender.com/auth/posts/model?type={n}"
        
        data = get_api_response(
            endpoint=api_url,
            timeout=15
        )
    
        for dat in data["data"]:
            post_datas = PostData(dat)
            post_data = post_datas.to_dict()
            print(post_data)
            post_manager = PostEmbeddingManager(
                post_file_path= POST_FILE_PATH,
                post_embeddings_file=EMBEDDING_FILE_PATH
            )
            post_manager.add_post(post_data)
        
        
    except APIError as ae:
        print(f"API Error: {str(ae)}")
    except Exception as e:
        print(f"General Error: {str(e)}")



@app.route('/recommend', methods=['POST'])
def recommend():
    
    user_data = {
        "Profession": data["occupation"],
        "Interests": data["location"] + " " +  data["bio"],
        "Interest_Categories": data["interest"],
        "interacted_posts": []
    }
    recommendations = recommender.recommend(user_data, use_faiss=False)