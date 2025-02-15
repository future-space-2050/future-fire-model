import requests
from flask import Flask, request, jsonify
from Recommender import Recommender
from ErrorMessage import RecommenderError
from DataProcessor import DataPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from Modeling import calculate_similarities
from User import User
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

data = pd.read_csv('./DataSet/User_profile.csv')

# External API URL to fetch user data (type is a dynamic query parameter)
backend_api_base_url = "https://future-fire-backend.onrender.com/auth/users/model"

# Function to fetch user information from the backend API


def get_user_data_from_backend(user_type):
    try:
        # Construct the full URL with the dynamic 'type' query parameter
        url = f"{backend_api_base_url}?type={user_type}"
        
        response = requests.get(url)
        print(response.json())
        if response.status_code == 200:
            return response.json()  # Return the JSON data from the API
        else:
            return None
    except Exception as e:
        raise RecommenderError(f"Error fetching user data from backend: {e}")

@app.route('/recommendations', methods=['GET'])
def recommendation():
    try:
        # Get the 'type' query parameter from the URL, with a default value of 4 (all users)
        user_type = request.args.get('type', default=4, type=int)
        
        # Fetch user data from the backend API based on the 'type' parameter
        user_data_list = get_user_data_from_backend(user_type)
        
        if not user_data_list or len(user_data_list) == 0:
            return jsonify({"error": "No user data available from backend"}), 404
        
        # Assuming we're working with the first user from the response (you can adjust as needed)
        user_data = user_data_list[0]
        
        # Create a user object with the fetched data
        user = User(user_data=user_data)

        # Get query parameters (e.g., top_n recommendations)
        top_n = request.args.get('top_n', default=10, type=int)

        # Convert user data into DataFrame for processing
        user_df = user.to_data_frame()
        
        # Initialize DataProcessor for data preprocessing
        data_preprocessor = DataPreprocessor(data, user_df)
        preprocessed_data = data_preprocessor.preprocess()

        # Calculate cosine similarity between users
        cosine_similarities = calculate_similarities(preprocessed_data, data_preprocessor.get_user_index())
        
        # Create a Recommender object
        recommender = Recommender(preprocessed_data, cosine_similarities, top=10)
        
        # Get the top recommended user IDs
        recommended_ids = recommender.get_recommendations()
        
        # Fetch the recommended user profiles from the dataset
        recommended_profiles = data[data['User_ID'].isin(recommended_ids)]
        
        # Convert the profiles to JSON format
        recommendations_json = recommended_profiles.to_dict(orient='records')
        
        # Return recommendations as JSON
        return jsonify(recommendations_json)

    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error getting recommendations: {e}"
        return jsonify({"error": error_message}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
