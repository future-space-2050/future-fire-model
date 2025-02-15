import requests
from flask import Flask, request, jsonify
from postData import PostData
from CosineSimilarityRecommender import *
from PostEmbeddingManager import *


app = Flask(__name__)

post_manager = PostEmbeddingManager(
    post_file_path=POST_FILE_PATH,
    post_embeddings_file=EMBEDDING_FILE_PATH
)

# New post manager

recommender = CosineSimilarityRecommender(post_manager)

@app.route('/add_post', methods=['POST'])
def add_post():
    data = request.get_json()
    
    new_post = PostData(
        title=data['title'],
        content=data['content'],
        category=data['category'],
    ).to_dict()
    
    post_manager.add_post(new_post)
    return jsonify({'message': 'Post added successfully'})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    new_post = PostData(
        title=data['title'],
        content=data['content'],
    ).to_dict()
    
    recommendations = recommender.recommend(new_post, use_faiss=False)
    
    return jsonify(recommendations)
