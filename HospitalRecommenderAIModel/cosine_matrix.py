import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from scipy import sparse
# Load dataset
dataset_path = "Ethiopian_Hospitals_Dataset.csv"
df = pd.read_csv(dataset_path)

# Handle NaN values (replace with empty string or fill accordingly)
df[['Region','City','Specialties', 'Services','Disease']] = df[['Region', 'City', 'Specialties', 'Services','Disease']].fillna('')

# Convert text columns to TF-IDF vectors
tfidfR = TfidfVectorizer()
tfidfC = TfidfVectorizer()
tfidfSp = TfidfVectorizer()
tfidfSv = TfidfVectorizer()
tfidfD = TfidfVectorizer()

region_tfidf = tfidfR.fit_transform(df['Region'])
specialties_tfidf = tfidfSp.fit_transform(df['Specialties'])
services_tfidf = tfidfSv.fit_transform(df['Services'])
cities_tfidf = tfidfC.fit_transform(df['City'])
disease_tfidf = tfidfD.fit_transform(df['Disease'])


# Convert latitude and longitude to numpy arrays and reshape
coords = df[['Latitude', 'Longitude']].fillna(0).to_numpy()

# Combine all features horizontally

# Normalize numerical coordinates before combining
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# Convert to sparse matrix for hstack
coords_sparse = sparse.csr_matrix(coords_scaled)

combined_features = hstack([region_tfidf, cities_tfidf, specialties_tfidf, services_tfidf, disease_tfidf, coords_sparse])

# Compute cosine similarity
cosine_sim = cosine_similarity(combined_features)

# Save the similarity matrix
with open("Hospitals_Model.pkl", "wb") as file:
    pickle.dump(cosine_sim, file)

print("Hospitals similarity matrix saved as Hospitals_Model.pkl")
