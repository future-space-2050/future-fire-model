from Modeling import *

from flask import Flask, request, jsonify
import json
import urllib.parse

app = Flask(__name__)

# Existing recommendation system code here (keep all your existing classes and functions)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    try:
        # Get URL-encoded JSON string from query parameter
        encoded_user_data = request.args.get('user_data')
        if not encoded_user_data:
            return jsonify({'error': 'Missing user_data parameter'}), 400
            
        # Decode and parse JSON
        user_data = json.loads(urllib.parse.unquote(encoded_user_data))
        
        # Get recommendations
        recommendations = runner(user_data)
        return jsonify(recommendations)
    
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format in user_data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
        
        


