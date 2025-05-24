from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("Ethiopian_Hospitals_Dataset.csv")
df[['Region', 'City', 'Specialties', 'Services', 'Disease']] = df[[
    'Region', 'City', 'Specialties', 'Services', 'Disease'
]].fillna('')
df['combined'] = df['Region'] + ' ' + df['City'] + ' ' + df['Specialties'] + ' ' + df['Services'] + ' ' + df['Disease']

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Recommendation function
def recommend_hospitals(query, top_n=5):
    user_vec = tfidf.transform([query])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Hospital_Name', 'Region', 'City', 'Services']]

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    if request.method == 'POST':
        user_input = request.form['query']
        results = recommend_hospitals(user_input).to_dict(orient='records')
    return render_template('index2.html', results=results)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
