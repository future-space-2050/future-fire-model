class DataPreprocessorError(Exception):
    """Custom exception for data preprocessing errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"DataPreprocessorError: {self.message}"

    def __repr__(self):
        return f"DataPreprocessorError('{self.message}')"
    
    
class RecommenderError(Exception):
    """Custom exception for recommender system errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"RecommenderError: {self.message}"
    
    def __repr__(self):
        return f"RecommenderError('{self.message}')"
    
    
class JsonToDataFrameError(Exception):
    """Custom exception for JSON to DataFrame conversion errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"JsonToDataFrameError: {self.message}"

    def __repr__(self):
        return f"JsonToDataFrameError('{self.message}')"
    
class ModelingError(Exception):
    """Custom exception for modeling errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"ModelingError: {self.message}"
    
    def __repr__(self):
        return f"ModelingError('{self.message}')"
    