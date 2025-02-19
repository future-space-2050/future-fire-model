from datetime import datetime
import pandas as pd
import logging

USER_FILE_PATH = r"Who-To-Follow\DataSet\User_profile.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User:
    _coordinates = pd.read_csv(r'who_to_follow\DataSet\coordinates.csv')
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
        print("user id")
        self._set_get_user_id()
        print("name")
        self._set_get_name()
        print("birth date")
        self._set_get_birth_date()
        print("city")
        self._set_get_city()
        print("profession")
        self._set_get_profession()
        print("age")
        self._calculate_age()
        print("location")
        self.__location_mapping()
        print("interests")
        self._set_get_interests()
        print("interest categories")
        self._set_get_interest_categories()
        print("BOOOM")
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
        self.interest_categories = self.user_data.get("interest_categories", "") or "Unknown"
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
            # Ensure that city is not None before calling .strip()
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
        
    def to_dataframe(self):
        return pd.DataFrame([self.to_dict()])
    
    
    def save(self):
        try:
            df = self.to_dataframe()  # Convert the object to a dataframe
            # Check if the user data already exists in the CSV file
            if df['user_id'].isin(pd.read_csv(USER_FILE_PATH)['user_id']).any():
                # Append the new data if the user is not already in the CSV file
                df.to_csv(USER_FILE_PATH, mode='a', header=not pd.read_csv(USER_FILE_PATH).empty, index=False)
                logger.info(f"User data saved to {USER_FILE_PATH}")
            else:
                logger.info("User data already exists in the file. Skipping save.")
        except Exception as e:
            logger.error(f"Error saving user data: {str(e)}")


    def __repr__(self):
        return f"<User {self.user_id}: {self.name} ({self.age}y) from {self.city}>"
    
    