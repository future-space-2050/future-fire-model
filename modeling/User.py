# Json User information to DataFrame
from ErrorMessage import JsonToDataFrameError

# Data Processing

import pandas as pd
import json

class User:
    def __init__(self, user_data):
        self.user_data = user_data
        self.user_df = None
        self.user_id = None
        self.user_index = None
    
    def to_data_frame(self):
        """Convert JSON user data to a DataFrame."""
        try:
            self.user_df = pd.DataFrame([self.user_data])
            self.user_id = self.user_df["User_ID"].values[0]
            return self.user_df
        except Exception as e:
            raise JsonToDataFrameError(f"Error converting JSON to DataFrame: {e}")