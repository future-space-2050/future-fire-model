from datetime import datetime
import os
import pandas as pd
import logging

USER_FILE_PATH = r"who_to_follow/DataSet/User_profile.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User:
    _coordinates = pd.read_csv(r'who_to_follow/DataSet/coordinates.csv')
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

        
        self._set_get_user_id()
        self._set_get_name()
        self._set_get_birth_date()
        self._set_get_city()
        self._set_get_profession()
        self._calculate_age()
        self.__location_mapping()
        self._set_get_interests()
        self._set_get_interest_categories()
       
        # self.save()
        


    def _set_get_user_id(self):
        self.user_id = self.user_data.get("userID") 
        logger.debug(f"User ID set to: {self.user_id}")



    def _set_get_name(self):
        self.name = self.user_data.get("fullName", "").strip() or "Anonymous"
        logger.debug(f"Name set to: {self.name}")

    def _set_get_interests(self):
        self.interests = self.user_data.get("interests", "") or "Unknown"
        logger.debug(f"Interests set to: {self.interests}")

    def _set_get_interest_categories(self):
        self.interest_categories = self.user_data.get("interestCategories", "") or "Unknown"
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
        self.city = self.user_data.get("location", "Unknown").strip()  
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
        print("Location")
        if not self.city:
            logger.info("No city specified for location mapping")
            self.latitude = 8
            self.longitude = 38
            return

        try:
            print("Mapping")
            city_clean = self.city.strip().lower() if self.city and isinstance(self.city, str) else None
            print(city_clean)
            
            if city_clean:
                city_coordinates = self._coordinates[
                    self._coordinates['city'].str.strip().str.lower() == city_clean
                ]

                if not city_coordinates.empty:
                    self.latitude = city_coordinates.iloc[0]['Latitude']
                    self.longitude = city_coordinates.iloc[0]['Longitude']
                    logger.info(f"Mapped coordinates for {self.city}: {self.latitude}, {self.longitude}")
                else:
                    logger.warning(f"No coordinates found for city: {self.city}")
                    self.latitude = 8
                    self.longitude = 38
            else:
                logger.warning(f"City value is invalid: {self.city}")
                self.latitude = 8
                self.longitude = 38
        except Exception as e:
            logger.error(f"Error in location mapping: {str(e)}")
            self.latitude = 8
            self.longitude = 38


    def to_dict(self):
        return {
            "userID": self.user_id,
            "fullName": self.name,
            "dateOfBirth": self.birth_date.isoformat() if self.birth_date else None,
            "age": self.age,
            "location": self.city,
            "occupation": self.profession,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "interests": self.interests,
            "interestCategories": self.interest_categories
        }
        
    def to_dataframe(self):
        return pd.DataFrame([self.to_dict()])
    

    def save(self):
        try:
            df = self.to_dataframe()
            
            if os.path.exists(USER_FILE_PATH):
                existing_df = pd.read_csv(USER_FILE_PATH)
                
                if self.user_id in existing_df["userID"].values:
                    existing_df.loc[existing_df['UserID'] == self.user_id, :] = df.values[0]
                    logger.info(f"User data updated for user ID: {self.user_id}")
                else:
                    existing_df = pd.concat([existing_df, df], ignore_index=True)
                    logger.info(f"User data added for user ID: {self.user_id}")
                
                existing_df.to_csv(USER_FILE_PATH, index=False)
            else:
                df.to_csv(USER_FILE_PATH, mode='w', header=True, index=False)
                logger.info(f"User data saved to {USER_FILE_PATH}")
        
        except Exception as e:
            logger.error(f"Error saving user data: {str(e)}")


    def __repr__(self):
        return f"<User {self.user_id}: {self.name} ({self.age}y) from {self.city}>"