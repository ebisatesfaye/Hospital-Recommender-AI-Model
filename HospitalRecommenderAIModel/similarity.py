import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from scipy import sparse
# Load dataset
dataset_path = "datasets/Ethiopian_Hospitals_Dataset.csv"
df = pd.read_csv(dataset_path)

# Handle NaN values (replace with empty string or fill accordingly)
df[['Region','City','Specialties', 'Services']] = df[['Region', 'City', 'Specialties', 'Services']].fillna('')

# Convert text columns to TF-IDF vectors
tfidfR = TfidfVectorizer()
tfidfC = TfidfVectorizer()
tfidfSp = TfidfVectorizer()
tfidfSv = TfidfVectorizer()


region_tfidf = tfidfR.fit_transform(df['Region'])
specialties_tfidf = tfidfSp.fit_transform(df['Specialties'])
services_tfidf = tfidfSv.fit_transform(df['Services'])
cities_tfidf = tfidfC.fit_transform(df['City'])
# Convert latitude and longitude to numpy arrays and reshape
coords = df[['Latitude', 'Longitude']].fillna(0).to_numpy()

# Combine all features horizontally

# Normalize numerical coordinates before combining
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# Convert to sparse matrix for hstack
coords_sparse = sparse.csr_matrix(coords_scaled)

combined_features = hstack([region_tfidf, cities_tfidf, specialties_tfidf, services_tfidf, coords_sparse])

# Compute cosine similarity
cosine_sim = cosine_similarity(combined_features)

# Save the similarity matrix
with open("trained_models/Hospitals_similarity.pkl", "wb") as file:
    pickle.dump(cosine_sim, file)

print("Hospitals similarity matrix saved as Hospitals_similarity.pkl")








#==========================================================================
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity

# # Load your dataset
# dataset_path = "Ethiopian_Hospitals_Dataset.csv"  # Replace with your actual dataset path
# df = pd.read_csv(dataset_path)

# # Assuming your dataset contains numerical values for similarity calculation
# # Convert it to a NumPy array
# data_matrix = df[['Region', 'Latitude', 'Longitude','Specialties','Services']].values  # Adjust column selection based on your dataset structure

# # Compute cosine similarity
# cosine_sim = cosine_similarity(data_matrix)

# # Save the similarity matrix as a .pkl file
# with open("Hospitals_similarity.pkl", "wb") as file:
#     pickle.dump(cosine_sim, file)

# print("Hospitals_Similarity matrix saved as similarity.pkl")
