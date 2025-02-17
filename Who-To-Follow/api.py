from json import JSONDecodeError
from venv import logger
import requests
from flask import Flask, request, jsonify
from Recommender import Recommender
from ErrorMessage import RecommenderError
from DataProcessor import DataPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from Modeling import calculate_similarities
import panda as pd
from requests import *
import threading
from User import User, SaveUser

USER_FILE_PATH = r"./DataSet/User_profile.csv"

app = Flask(__name__)

backend_api_base_url = "https://future-fire-backend.onrender.com/auth/users/model"



def get_user_data_from_backend(user_type):
    try:
        url = f"{backend_api_base_url}?type={user_type}"
        
        response = requests.get(url)
        print(response.json())
        if response.status_code == 200:
            return response.json() 
        else:
            return None
    except Exception as e:
        raise RecommenderError(f"Error fetching user data from backend: {e}")

@app.route('/recommendations', methods=['GET'])
def recommendation():
    try:
        user_type = request.args.get('type', default=4, type=int)
        
        user_data_list = get_user_data_from_backend(user_type)
        
        if not user_data_list or len(user_data_list) == 0:
            return jsonify({"error": "No user data available from backend"}), 404
       
        user_data = user_data_list[0]

        user = User(user_data=user_data)

        top_n = request.args.get('top_n', default=10, type=int)

        user_df = user.to_data_frame()
        data_preprocessor = DataPreprocessor(data, user_df)
        preprocessed_data = data_preprocessor.preprocess()
        cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
        recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
        
        recommended_ids = recommender.get_recommendations()
        recommended_profiles = data[data['User_ID'].isin(recommended_ids)]
        recommendations_json = recommended_profiles.to_dict(orient='records')
        
        return jsonify(recommendations_json)
    except Exception as e:
        error_message = f"Error getting recommendations: {e}"
        return jsonify({"error": error_message}), 500
    
def get_api_user_response(endpoint, params=None, headers=None, timeout=60):

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

    except requests.Timeout as te:
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


if __name__ == "__main__":
   
    try:
        api_url = "https://future-fire-backend.onrender.com/auth/users/model?type=4"
        data = get_api_user_response(api_url, timeout=20)
        
        if data and "data" in data:
            for user_info in data["data"]:
                user = User(user_info)
                print(user.to_dict())
                
    except APIError as ae:
        logger.error(f"API Error: {str(ae)}")
    except Exception as e:
        logger.error(f"General Error: {str(e)}")

    app.run(debug=True)