from venv import logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

from who_to_follow.error_message import *

class DataPreprocessor:
    def __init__(self, data, user):
        self.data = data
        self.user = user
        self.user_index = None

    def list_to_string(self, lst):
        """Convert a list or string representation of a list into a cleaned string."""
        try:
            if isinstance(lst, str):
                result = lst.replace("[", "").replace("]", "").replace("'", "").replace(" & ", "_")
            elif isinstance(lst, list):
                result = ", ".join(lst)
            else:
                result = ""
            return result
        except Exception as e:
            raise DataPreprocessorError(f"Error converting list to string: {e}")

    def one_hot_encode(self):
        """One-hot encode categorical features like Interests, Interest Categories, City, Gender, and Profession."""
        try:
            self.data["Interests"] = self.data["Interests"].apply(self.list_to_string)
            self.data["Interest_Categories"] = self.data["Interest_Categories"].apply(self.list_to_string)

            self.data["Interests"] = self.data["Interests"].str.split(", ")
            self.data["Interest_Categories"] = self.data["Interest_Categories"].str.split(", ")

            mlb_interests = MultiLabelBinarizer()
            mlb_categories = MultiLabelBinarizer()
            interests_encoded = pd.DataFrame(
                mlb_interests.fit_transform(self.data["Interests"]),
                columns=[f"Interest_{x}" for x in mlb_interests.classes_],
                index=self.data.index
            )
            categories_encoded = pd.DataFrame(
                mlb_categories.fit_transform(self.data["Interest_Categories"]),
                columns=[f"Category_{x}" for x in mlb_categories.classes_],
                index=self.data.index
            )

            self.data = self.data.drop(columns=["Interests", "Interest_Categories"])
            self.data = pd.concat([self.data, interests_encoded, categories_encoded], axis=1)

            self.data = pd.get_dummies(self.data, columns=["City", "Profession"], drop_first=True)
            return self.data
        except Exception as e:
            raise DataPreprocessorError(f"Error during one-hot encoding: {e}")

    def drop_columns(self):
        """Drop unnecessary columns like User ID, Name, and Birth Date."""
        try:
            self.data = self.data.drop(columns=[ 'Name', 'Birth_Date', "Gender"])
            return self.data
        except Exception as e:
            raise DataPreprocessorError(f"Error during column dropping: {e}")

    def normalize(self):
        """Normalize numerical features like Latitude, Longitude, and Age."""
        try:
            scaler = MinMaxScaler()
            numerical_columns = ["Latitude", "Longitude", "Age"]
            self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

            self.data["distance"] = 1 - self.data["distance"]
            return self.data
        except Exception as e:
            raise DataPreprocessorError(f"Error during normalization: {e}")

    def add_user(self):
        """Add the target user to the dataset if not already present."""
        try:
            existing_index = self.is_in_data()
            if existing_index is not None:
                self.user_index = existing_index
            else:
                user_df = self.user.copy()
                missing_cols = set(self.data.columns) - set(user_df.columns)
                for col in missing_cols:
                    user_df[col] = 0 
                user_df = user_df[self.data.columns]
                self.data = pd.concat([self.data, user_df], ignore_index=True)
                self.user_index = self.data.index[-1]
            return self.data
        except Exception as e:
            raise DataPreprocessorError(f"Error adding user: {e}")


    def distance_from_user(self):
        """Compute the Euclidean distance between the target user and all other users."""
        try:
            print("Distance from user")
            print("Data shape", self.user.shape)
            self.data["distance"] = 0
            
            user_lat = self.user.iloc[0]["Latitude"]
            user_long = self.user.iloc[0]["Longitude"]

            print("Distance", user_lat, user_long)
            self.data["distance"] = (self.data["Latitude"] - user_lat) ** 2 + (self.data["Longitude"] - user_long) ** 2
            self.data["distance"] = self.data["distance"] ** 0.5
            return self.data
            
        except Exception as e:
            raise DataPreprocessorError(f"Error computing distance from user: {e}")

    def is_in_data(self):
        """
        Checks if the user exists in the dataset and returns their index if found.
        
        Returns:
            int or None: Index of the user in the dataset, or None if not found.
        
        Raises:
            DataPreprocessorError: If there's an error during the check.
        """
        try:
            # Check if user data is empty
            if self.user.empty:
                logger.warning("User data is empty.")
                return None
            
            # Extract user ID safely
            user_id = self.user["User_ID"].iloc[0]  # Use iloc to avoid SettingWithCopyWarning
            
            # Check if the user exists in the dataset
            user_exists = self.data["User_ID"].eq(user_id).any()
            
            if user_exists:
                # Get the first occurrence's index (assuming User_IDs are unique)
                user_index = self.data.index[self.data["User_ID"] == user_id].tolist()[0]
                return user_index
            else:
                return None
        except IndexError:
            logger.error(f"User ID {user_id} not found in the dataset.")
            return None
        except Exception as e:
            raise DataPreprocessorError(f"Error checking for user in data: {str(e)}")

    

    def preprocess(self):
        try:
            self.data = self.add_user()
            self.data = self.data.drop(columns=["User_ID"])
            self.data = self.distance_from_user()
            self.data = self.one_hot_encode()
            self.data = self.drop_columns()
            self.data = self.normalize()
            # Check for NaN values
            if self.data.isna().any().any():
                raise DataPreprocessorError("NaN values detected after preprocessing.")
            return self.data
        except Exception as e:
            raise DataPreprocessorError(f"Error during preprocessing: {e}")

    def get_user_index(self):
        """Get the index of the target user in the preprocessed data."""
        return self.user_index
