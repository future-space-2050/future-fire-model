from .error_message import RecommenderError

class Recommender:
    def __init__(self, data, cosine_sim_model, user_index=None, top=3):
        self.data = data
        self.cosine_sim_model = cosine_sim_model
        self.top = top
        self.user_index = user_index if user_index is not None else len(data) - 1

    def get_recommendations(self):
        """Get top-N recommendations based on cosine similarity."""
        try:
            sim_vectors = list(enumerate(self.cosine_sim_model))
            sim_vectors = sorted(sim_vectors, key=lambda x: x[1], reverse=True)
            sim_vectors = sim_vectors[1:self.top + 1]
            user_indices = [i[0] for i in sim_vectors]
            return user_indices
        except Exception as e:
            raise RecommenderError(f"Error getting recommendations: {e}")