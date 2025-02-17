from datetime import datetime
import pandas as pd
import logging

USER_FILE_PATH = r"Who-To-Follow\DataSet\User_profile.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User:
    _coordinates = pd.read_csv(r'Who-To-Follow/DataSet/coordinates.csv')
    def __init__(self, user_data):
        self.user_data = user_data
        self.user_id = None
        self.name = None
        self.birth_date = None
        self.city = None
        self.profession = None
        self.age = None
        self.latitude = None
        self.longitude = None
        self.interests = None
        self.interest_categories = None

        # Initialize attributes
        self._set_get_user_id()
        self._set_get_name()
        self._set_get_birth_date()
        self._set_get_city()
        self._set_get_profession()
        self._calculate_age()
        self.__location_mapping()
        self._set_get_interests()
        self._set_get_interest_categories()

    def _set_get_user_id(self):
        self.user_id = self.user_data.get("userID")
        logger.debug(f"User ID set to: {self.user_id}")

    

    def _set_get_name(self):
        self.name = self.user_data.get("fullName", "").strip() or "Anonymous"
        logger.debug(f"Name set to: {self.name}")

    def _set_get_interests(self):
        self.interests = self.user_data.get("interests", "").strip() or "Unknown"
        logger.debug(f"Interests set to: {self.interests}")

    def _set_get_interest_categories(self):
        self.interest_categories = self.user_data.get("interest_categories", "").strip() or "Unknown"
        logger.debug(f"Interest categories set to: {self.interest_categories}")

    def _set_get_birth_date(self):
        dob_str = self.user_data.get("dateOfBirth")
        if dob_str and dob_str.strip():
            try:
                self.birth_date = datetime.strptime(dob_str, "%Y-%m-%d").date()
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid birth date format: {dob_str} - {str(e)}")
                self.birth_date = None
        else:
            self.birth_date = None
        logger.debug(f"Birth date set to: {self.birth_date}")

    def _set_get_city(self):
        self.city = self.user_data.get("location", "").strip() or None
        logger.debug(f"City set to: {self.city}")

    def _set_get_profession(self):
        self.profession = self.user_data.get("occupation", "").strip() or "Unknown"
        logger.debug(f"Profession set to: {self.profession}")

    def _calculate_age(self):
        if self.birth_date:
            today = datetime.today()
            self.age = today.year - self.birth_date.year - (
                (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
            )
        else:
            self.age = None
        logger.debug(f"Age calculated as: {self.age}")

    def __location_mapping(self):
        if not self.city:
            logger.info("No city specified for location mapping")
            self.latitude = None
            self.longitude = None
            return

        try:
            city_clean = self.city.strip().lower()
            city_coordinates = self._coordinates[
                self._coordinates['city'].str.strip().str.lower() == city_clean
            ]
            
            if not city_coordinates.empty:
                self.latitude = city_coordinates.iloc[0]['Latitude']
                self.longitude = city_coordinates.iloc[0]['Longitude']
                logger.info(f"Mapped coordinates for {self.city}: {self.latitude}, {self.longitude}")
            else:
                logger.warning(f"No coordinates found for city: {self.city}")
                self.latitude = None
                self.longitude = None
        except Exception as e:
            logger.error(f"Error in location mapping: {str(e)}")
            self.latitude = None
            self.longitude = None

    def to_dict(self):
        return {
            "User_ID": self.user_id,
            "Name": self.name,
            "Birth_Date": self.birth_date.isoformat() if self.birth_date else None,
            "Age": self.age,
            "City": self.city,
            "Profession": self.profession,
            "Latitude": self.latitude,
            "Longitude": self.longitude,
            "Interests": self.interests,
            "Interest_Categories": self.interest_categories
        }

    def __repr__(self):
        return f"<User {self.user_id}: {self.name} ({self.age}y) from {self.city}>"


class SaveUser:
    def __init__(self, user_data):
        self.user_data = user_data
        self.user = User(user_data=user_data)
        
    def _save_to_dataset(self, file_path = USER_FILE_PATH):
        try:
            user_df = pd.read_csv(file_path)
            if self.find_user_by_id(user_id=self.user["User_ID"]):
                user_df = user_df.append(self.user.to_dict(), ignore_index=True)
                user_df.to_csv(file_path, index=False)
                logger.info(f"User data saved to {file_path}")
            else:
                logger.info(f"User data already exists for User ID: {self.user['User_ID']}")
        except Exception as e:
            logger.error(f"Error in saving user data: {str(e)}")
            
            
    def save_user_data(self):
        self._save_to_dataset()
        
    def get_user_data(self):
        return self.user.to_dict()
    
    def find_user_by_id(self, user_id):
        try:
            user_df = pd.read_csv(USER_FILE_PATH)
            
            user_data = user_df[user_df['User_ID'] == user_id].to_dict(orient='records')
            if user_data:
                return User(user_data[0])
            else:
                logger.info(f"No user found with User ID: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error in finding user by ID: {str(e)}")
            return None