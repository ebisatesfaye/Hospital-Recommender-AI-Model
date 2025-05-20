import pandas as pd
import pickle


dataset_path = "Ethiopian_Hospitals_Dataset.csv"  # Replace with your actual dataset path

data = pd.read_csv(dataset_path)
# Define column names
# columns = ["id", "title", "genre", "original_language", "popularity", "release_date", "vote_average", "vote_count"]

# Create DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as a pickle file
with open("Hospitals_list.pkl", "wb") as file:
    pickle.dump(df, file)

print("Hospitals_list.pkl file created successfully!")
