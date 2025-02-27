import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from post_recommender.CosineSimilarityRecommender import *


class PostEmbeddingManager:
    def __init__(self, post_file_path=POST_FILE_PATH, post_embeddings_file=EMBEDDING_FILE_PATH):
        self.post_file_path = post_file_path
        self.post_embeddings_file = post_embeddings_file
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not os.path.exists(self.post_file_path):
            pd.DataFrame(columns=['post_id', 'category', 'content']).to_csv(self.post_file_path, index=False)
        
        if os.path.exists(self.post_embeddings_file):
            embeddings = np.load(self.post_embeddings_file, allow_pickle=False)
            if embeddings.shape[0] == 0:
                print("Embeddings file is empty - waiting for first entries")
        else:
            embeddings = np.array([
                self._generate_embedding(post['category'] if post['category'] else " " + " " + post['content'] if post['content'] else " ")
                for post in pd.read_csv(self.post_file_path).to_dict('records')
            ])
            # save 
            np.save(self.post_embeddings_file, embeddings)
        

    def _generate_embedding(self, text):
        """Generate embedding for a single text input"""
        return self.embedder.encode(
            text, 
            convert_to_numpy=True
        ).astype(np.float32)

    def add_post(self, new_post):
        """
        Add new post and update embeddings
        Args:
            new_post (dict): Should contain 'post_id', 'category', and 'content'
        """
        if not self.__is_post_exist(new_post=new_post):
            if not all(key in new_post for key in ['post_id', 'category', 'content']):
                raise ValueError("New post must contain post_id, category, and content")

            self._append_to_csv(new_post)
            self._update_embeddings(new_post)
            return True
        else:
            print("Post already exists")
            return False

    def __is_post_exist(self, new_post):
        try:
            posts_df = pd.read_csv(
                self.post_file_path, 
                dtype={'post_id': str} 
            )
        
            exists = posts_df['post_id'].str.strip().isin([str(new_post['post_id']).strip()]).any()

            print(f"Post {new_post['post_id']} exists: {exists}")
            return exists
            
        except FileNotFoundError:
            print("Post file not found")
            return False
        except KeyError:
            print("'post_id' column missing")
            return False

    def _append_to_csv(self, new_post):
        """Append new post to the CSV file"""
        new_post_df = pd.DataFrame([new_post])
        
        if os.path.getsize(self.post_file_path) > 0:
            existing_df = pd.read_csv(self.post_file_path)
            updated_df = pd.concat([existing_df, new_post_df], ignore_index=True)
        else:
            updated_df = new_post_df
            
        updated_df.to_csv(self.post_file_path, index=False)

    def _update_embeddings(self, new_post):
        """Update embeddings with new post"""
        combined_text = f"{new_post['category']} {new_post['content']}"
        new_embedding = self._generate_embedding(combined_text)
        
        existing_embeddings = np.load(self.post_embeddings_file)
       
        updated_embeddings = np.vstack([existing_embeddings, new_embedding])
        
        np.save(self.post_embeddings_file, updated_embeddings)


    def get_latest_embeddings(self):
        """Return current embeddings array"""
        return np.load(self.post_embeddings_file)
    
    def get_dataframe(self):
        return self.posts_df