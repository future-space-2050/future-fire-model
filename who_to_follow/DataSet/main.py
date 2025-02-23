import pandas as pd

USER_FILE_PATH = r"User_profile.csv"
user_data = pd.read_csv(USER_FILE_PATH)

# Read data from the user profile and save the 250 data rows

top = 0 

while top < len(user_data):
    user_data_top = user_data.iloc[top:top + 250]
    user_data_top.to_csv(f"user_profile_data_{(top // 250) + 1}.csv", index=False)
    
    top += 250
    print(f"Saved user_profile_data_{top // 250}.csv")
    
