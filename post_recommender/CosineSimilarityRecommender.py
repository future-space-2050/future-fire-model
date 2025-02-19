import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity


POST_FILE_PATH = r"post_recommender\Post Dataset\POST_DATASET.csv"
EMBEDDING_FILE_PATH = r"post_recommender\Post Dataset\POST_EMBEDDINGS.npy"
USERS_FILE_PATH = r"who_to_follow\DataSet\User_profile.csv"


class CosineSimilarityRecommender:
    def __init__(self, posts_df, user_data=None):
        # Preserve original post IDs and indices
        self.posts_df = posts_df
        
          # Store original IDs
        
        self.user_data = user_data
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.post_embeddings = self._get_cached_embeddings()
        self.post_embeddings = self._normalize(self.post_embeddings)
        
        self.index = faiss.IndexFlatIP(self.post_embeddings.shape[1])
        self.index.add(self.post_embeddings.astype(np.float32))

    def _get_cached_embeddings(self, emb_file=EMBEDDING_FILE_PATH):
        """Smart embedding cache with validation"""
        if os.path.exists(emb_file):
            cached = np.load(emb_file)
            print(type(self.posts_df))
            if len(cached) == self.posts_df.shape[0]:
                return cached.astype(np.float32)
        
        if 'combined_text' not in self.posts_df.columns:
            self.posts_df['combined_text'] = (
                self.posts_df['category'] + " " + 
                self.posts_df['content']
            )
            
        embeddings = self.embedder.encode(
            self.posts_df['combined_text'],
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        np.save(emb_file, embeddings)
        return embeddings

    def _normalize(self, vectors):
        """L2 normalization for cosine similarity"""
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def get_user_embedding(self):
        """Create normalized user embedding profile"""
        if self.user_data is None:
            return np.mean(self.post_embeddings, axis=0)
        
        user_data = self.user_data

        profile_text = f"{user_data['Profession']} {user_data['Interests']} {user_data['Interest_Categories']}"
        profile_emb = self.embedder.encode(profile_text)
        profile_emb = self._normalize(profile_emb.reshape(1, -1))[0]

        interacted = user_data.get('interacted_posts', [])
        valid_interactions = self.posts_df[
            self.posts_df['post_id'].isin(interacted)
        ].index.tolist()
        
        if valid_interactions:
            interaction_emb = np.mean(
                self.post_embeddings[valid_interactions], 
                axis=0
            )
        else:
            interaction_emb = np.zeros_like(profile_emb)
            
        combined = 0.4 * interaction_emb + 0.6 * profile_emb
        return self._normalize(combined.reshape(1, -1))[0]

    def recommend(self, top_n=5, use_faiss=True, batch_size=512, exclude_interacted=True):
        """
        Get recommendations using either FAISS or batch-wise cosine similarity
        
        Parameters:
        - user_id: ID of the user to recommend for
        - top_n: Number of recommendations to return
        - use_faiss: Use FAISS for accelerated search (default True)
        - batch_size: Batch size for non-FAISS processing
        - exclude_interacted: Filter out already interacted posts
        
        Returns: List of post IDs in recommendation order
        """
        try:
            user_emb = self.get_user_embedding()
            user_emb = user_emb.astype(np.float32).reshape(1, -1)
            
            interacted = set()
            if exclude_interacted and self.user_data is not None:
                if self.user_data:
                    interacted = set(self.user_data.get('interacted_posts', []))
            
            recommendations = []
            
            if use_faiss:
                _, indices = self.index.search(user_emb, top_n * 3)
                for idx in indices[0]:
                    if idx >= len(self.posts_df):
                        continue
                    post_id = self.posts_df.iloc[idx]['post_id']
                    if post_id not in interacted:
                        recommendations.append(post_id)
                        if len(recommendations) >= top_n:
                            break
            else:
                seen = set()
                for i in range(0, len(self.post_embeddings), batch_size):
                    batch_embs = self.post_embeddings[i:i+batch_size]
                    scores = cosine_similarity(user_emb, batch_embs)[0]
                    batch_ids = self.posts_df.iloc[i:i+batch_size]['post_id']
                    
                    for score, pid in zip(scores, batch_ids):
                        if pid not in seen and pid not in interacted:
                            recommendations.append((score, pid))
                            seen.add(pid)
                
                recommendations = [pid for _, pid in sorted(recommendations, reverse=True)]
            
            final_recs = []
            seen_pids = set()
            for pid in recommendations:
                if pid not in seen_pids:
                    final_recs.append(pid)
                    seen_pids.add(pid)
                if len(final_recs) >= top_n:
                    break
            
            return final_recs[:top_n]
        
        except KeyError:
            print(f"User {user_id} not found, returning popular posts")
            return self.posts_df.sample(top_n)['post_id'].tolist()
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return self.posts_df.sample(top_n)['post_id'].tolist()
        
        
    def __len__(self):
        return len(self.embeddings) 