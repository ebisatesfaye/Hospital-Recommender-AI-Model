from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
from difflib import get_close_matches
from flask import Flask, request, render_template, jsonify  # Import jsonify
import pandas as pd
import re
import ast


df = pd.read_csv("Ethiopian_Hospitals_Dataset.csv")
# Step 1: Preprocess the DataFrame
df[['Region', 'City', 'Specialties', 'Services', 'Disease']] = df[[
    'Region', 'City', 'Specialties', 'Services', 'Disease'
]].fillna('')

df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].fillna(0)

# Step 2: Define feature and target
X = df[['Region', 'City', 'Specialties', 'Services', 'Disease', 'Latitude', 'Longitude']]
y = df['Hospital_Name']

# Encode target labels (hospital names)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Build column transformer for preprocessing
text_features = ['Region', 'City', 'Specialties', 'Services', 'Disease']
numeric_features = ['Latitude', 'Longitude']

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), 'Region'),         # We'll repeat this for each text feature
    ('city', TfidfVectorizer(), 'City'),
    ('specialty', TfidfVectorizer(), 'Specialties'),
    ('service', TfidfVectorizer(), 'Services'),
    ('disease', TfidfVectorizer(), 'Disease'),
    ('num', StandardScaler(), ['Latitude', 'Longitude'])
])

# Step 4: Create pipeline with classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Step 5: Train the model
model_pipeline.fit(X, y_encoded)

# Step 6: Save the model and label encoder for later use
with open("hospital_recommender_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

with open("hospital_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


print("Model training complete. Files saved: 'hospital_recommender_model.pkl' and 'hospital_label_encoder.pkl'")