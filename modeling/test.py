import json
import requests
import urllib.parse

user_data = {
    "User_ID": 1,
    "Name": "Kiilolee Daawwitii",
    "City": "Muger",
    "Birth_Date": "2007-06-21",
    "Interests": ["Songwriting", "Filmmaking", "Sculpting", "Art", "Coaching"],
    "Gender": "Male",
    "Interest_Categories": ["Animals_Nature", "Arts_Creativity", "Science_Technology"],
    "Profession": "Digital Content Creator",
    "Latitude": 8.582099,
    "Longitude": 38.885129,
    "Birth_Year": 2007,
    "Age": 17
}

# URL-encode the JSON data
encoded_data = urllib.parse.quote(json.dumps(user_data))

response = requests.get(f'http://localhost:5000/recommend?user_data={encoded_data}')
print(response.json())