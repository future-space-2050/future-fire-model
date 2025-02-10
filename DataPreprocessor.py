from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import joblib
import ast
from pathlib import Path


class DataPreprocessor:
    def __init__(self, embeddings_path="user_embeddings.npy"):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = embeddings_path
        self.base_embeddings = None
        
        # Load existing embeddings if available
        if Path(embeddings_path).exists():
            self.base_embeddings = np.load(embeddings_path)

    def safe_parse_interests(self, x):
        """Standardize different interest formats"""
        try:
            if isinstance(x, str) and x.startswith('['):
                parsed = ast.literal_eval(x)
                return ' '.join(parsed) if isinstance(parsed, list) else str(parsed)
            return x.replace(';', ',').replace(', ', ',').replace(',', ' ')
        except (SyntaxError, ValueError):
            return x.replace(',', ' ').replace(';', ' ')


    def preprocess(self, data):
        """Main preprocessing pipeline"""
        data = data.copy()
        # Features Drop Cols
        data = self.drop_cols(data)

        preprocessed_features = ['Interests', 'Interest Categories', "Profession"]
        for feature in preprocessed_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna('').apply(self.safe_parse_interests)

        return data


    def drop_cols(self, data):
        """Clean and preprocess user data"""
        # Drop Gender 
        for col in ["Gender", "max_prob"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        return data

    
    def clean_user_data(self, user_data):
        """Clean and preprocess user data"""
        return self.preprocess(pd.DataFrame([user_data])).iloc[0]


    def compute_base_embeddings(self, data):
        """Compute and save embeddings for base users"""
        self.base_embeddings = self.sbert_model.encode(data['Interests'].fillna(""), convert_to_numpy=True)
        np.save(self.embeddings_path, self.base_embeddings)
        print(f"Saved base embeddings to {self.embeddings_path}")
        return self.base_embeddings

    def compute_new_user_embedding(self, user_data):
        """Compute embedding for a single new user"""
        return self.sbert_model.encode([user_data['Interests']], convert_to_numpy=True)[0]
