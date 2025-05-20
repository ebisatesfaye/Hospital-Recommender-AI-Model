from difflib import get_close_matches
from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import re
import ast

# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")
ethiopian_hospitals = pd.read_csv("Ethiopian_Hospitals_Dataset.csv")

with open("Hospitals_Model.pkl", "rb") as file:
    cosine_sim = pickle.load(file)
hospital_names = ethiopian_hospitals['Hospital_Name'].tolist()

def handle_user_input(user_input, ethiopian_hospitals, cosine_sim, hospital_names, top_n=5):
    # Step 1: Try to match hospital name directly
    for name in hospital_names:
        if user_input in name.lower():
            return recommend_hospitals(name, cosine_sim, hospital_names, top_n)

    # Step 2: Try matching Region
    region_matches = ethiopian_hospitals[ethiopian_hospitals['Region'].str.lower().str.contains(user_input)]
    if not region_matches.empty:
        hospital_name = region_matches.iloc[0]['Hospital_Name']
        return recommend_hospitals(hospital_name, cosine_sim, hospital_names, top_n)

    # Step 3: Try matching City
    city_matches = ethiopian_hospitals[ethiopian_hospitals['City'].str.lower().str.contains(user_input)]
    if not city_matches.empty:
        hospital_name = city_matches.iloc[0]['Hospital_Name']
        return recommend_hospitals(hospital_name, cosine_sim, hospital_names, top_n)

    # Step 4: Try matching Services
    services_matches = ethiopian_hospitals[ethiopian_hospitals['Services'].str.lower().str.contains(user_input)]
    if not services_matches.empty:
        hospital_name = services_matches.iloc[0]['Hospital_Name']
        return recommend_hospitals(hospital_name, cosine_sim, hospital_names, top_n)

    # Step 5: If no match found
    return ["No hospitals found matching your input."]

def handle_user_input_multi(user_input, df, cosine_sim, hospital_names, top_n=5):
    user_inputs = "user_input".strip().lower()
    list_of_keywords = user_input
    print("User Input:", user_input)
    matched_indices = []
 
    for keyword in list_of_keywords:
    # Match by hospital name
        for idx, name in enumerate(hospital_names):
            if keyword in name.lower():
                matched_indices.append(idx)

        # Match by Region
        region_matches = df[df['Region'].str.lower().str.contains(keyword)]
        matched_indices += region_matches.index.tolist()

        # Match by City
        city_matches = df[df['City'].str.lower().str.contains(keyword)]
        matched_indices += city_matches.index.tolist()

        # Match by Services
        services_matches = df[df['Services'].str.lower().str.contains(keyword)]
        matched_indices += services_matches.index.tolist()

        # Match by Specialties
        specialties_matches = df[df['Specialties'].str.lower().str.contains(keyword)]
        matched_indices += specialties_matches.index.tolist()

    # Remove duplicates and convert to set
    matched_indices = list(set(matched_indices))

    if not matched_indices:
        return ["No hospitals found matching your input."]

    # Aggregate similarity scores from all matched hospitals
    similarity_scores = sum([cosine_sim[i] for i in matched_indices])

    # Create list of (index, score), excluding matched hospitals themselves
    results = [(i, score) for i, score in enumerate(similarity_scores) if i not in matched_indices]

    # Sort by score descending
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Take top N and return hospital names
    top_indices = [i[0] for i in results[:top_n]]
    return [hospital_names[i] for i in top_indices]


# Sample list of known cities, services, diseases (could be expanded)
known_cities = ethiopian_hospitals['City'].unique().tolist()
known_services = ethiopian_hospitals['Services'].unique().tolist()
split_services = [service.strip() for entry in known_services for service in entry.split("|")]

unique_services = list(set(split_services))
unique_services.sort()  # Optional: sort alphabetically
print(unique_services)
known_Disease = ethiopian_hospitals['Disease'].unique().tolist()

def extract_keywords_from_input(text):
    text = text.lower()
    city_keywords = [city for city in known_cities if city.lower() in text]
    service_keywords = [s for s in unique_services if s.lower() in text]
    # Assuming no spaces around the "|"
 
    specialty_keywords = [s for s in known_Disease if s.lower() in text]

    # Combine all found keywords
    keywords = []
    keywords.extend(city_keywords)
    keywords.extend(service_keywords)
    keywords.extend(specialty_keywords)
    return keywords


def recommend_hospitals(hospital_name, cosine_sim, hospital_names, top_n=5):
    try:
        # Get the index of the hospital from the list
        index = hospital_names.index(hospital_name)

        # Get similarity scores for that hospital
        similarity_scores = list(enumerate(cosine_sim[index]))

        # Sort by similarity score in descending order
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Exclude the first one (it's the same hospital), then get top N
        top_indices = [i[0] for i in similarity_scores[1:top_n+1]]

        # Return the most similar hospital names
        return [hospital_names[i] for i in top_indices]

    except ValueError:
        return ["Hospital not found. Please check the name."]

 
# creating routes========================================
@app.route("/")
def index():
    return render_template("index.html")


# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['symptoms'].strip().lower()
        keywords = extract_keywords_from_input(user_input)
        if not keywords:
            return render_template("index.html", hospitals=["Sorry, I couldn't understand your request."])

        # Combine keywords into a single search string
        combined_query = " ".join("hi")
        print("Combined Query:", keywords)
        recommendations = handle_user_input_multi(keywords, ethiopian_hospitals, cosine_sim, hospital_names)
        print("Recommendations:", recommendations)
        return render_template('index.html', hospitals=recommendations)
    # hospital_name = request.form.get('symptoms')
    # # mysysms = request.form.get('mysysms')
    # # print(mysysms)
        # print(hospital_name)
        # # if symptoms not in symptoms_dict.keys():
        # if hospital_name =="Symptoms":
        #     message = "Please either write symptoms or you have written misspelled symptoms"
        #     return render_template('index.html', message=message)
        # else:
        #     # Split the user's input into a list of symptoms (assuming they are comma-separated)
        #     user_symptoms = [s.strip() for s in hospital_name.split(' ')]
        #     # Remove any extra characters, if any
        #     user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        #     print(user_symptoms)
        #     predicted_hospitals = recommend_hospitals(hospital_name, cosine_sim, hospital_names, top_n=5)
        #     # hospitals = []

        #     # hospital_recommendations = list(set(hospitals))
        #     # if hospitals:
        #     #     hospital_recommendations = hospitals[0]
        #     # else:
        #     #     hospital_recommendations = hospitals
        #     # print(hospital_recommendations)
        #     # dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        #     # medications_list = ast.literal_eval(medications[0])
        #     # my_precautions = []
        #     # for i in precautions[0]:
        #     #     my_precautions.append(i)

        #     return render_template('index.html', hospitals=predicted_hospitals)

    return render_template('index.html')

 
if __name__ == '__main__':
    app.run(debug=True)