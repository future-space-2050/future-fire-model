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

@app.route('/add_post', methods=['POST'])
def add_post():
    data = request.get_json()
    
    new_post = PostData(
        title=data['title'],
        content=data['content'],
        category=data['category'],
    )
    post_manager.add_post(new_post)
    
    return jsonify({"post_id": new_post.post_id}), 201

@app.route('/get_post', methods=['GET'])
def get_post():
    post_id = request.args.get('post_id')
    if not post_id:
        return jsonify({"error": "Missing post_id parameter"}), 400
    
    post = post_manager.get_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    return jsonify(post.to_dict())


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    user_data = {
        "Profession": data["occupation"],
        "Interests": data["location"] + " " +  data["bio"],
        "Interest_Categories": data["interest"],
        "interacted_posts": []
    }
    recommendations = recommender.recommend(user_data, use_faiss=False)
    
    return jsonify(recommendations)


def post_data_loading():
    post_data = pd.read_csv(
        POST_FILE_PATH,
        usecols=['post_id', 'content', 'category']
    )
    return post_data
