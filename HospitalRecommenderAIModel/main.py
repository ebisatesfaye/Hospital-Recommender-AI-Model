from difflib import get_close_matches
from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
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

# Select specific columns, e.g., 'Name', 'Region', 'Type'
# selected_columns = ethiopian_hospitals[['Name', 'Region', 'Type']]

# Convert to list of dictionaries
# list_of_dicts = selected_columns.to_dict(orient='records')
# load model===========================================
svc = pickle.load(open('svc.pkl','rb'))

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
regions_dict = {'Afar': 0, 'Amhara': 1, 'Benishangul-Gumuz': 2, 'Dire Dawa': 3, 'Gambela': 4, 'Harari': 5, 'Oromia': 6, 'Somali': 7, 'South Ethiopia Regional StateR': 8, 'Tigray': 9 , 'Southwest Ethiopia Regional State':10, 'Addis Ababa':11,'Central Ethiopia Regional State':12,'Sidama':13}

city_list = ethiopian_hospitals[['City']].to_dict(orient='records')
cities_list = list({item['City'] for item in city_list})

Service_dict_list = ethiopian_hospitals[['Services']].to_dict(orient='records')
Services_list = list({item['Services'] for item in Service_dict_list})

specialty_dict_list = ethiopian_hospitals[['Specialties']].to_dict(orient='records')
Specialties_list = list({item['Specialties'] for item in specialty_dict_list})



def get_predicted_value(patient_symptoms): 
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            # Try to find the closest symptom using fuzzy matching
            closest_matches = get_close_matches(symptom, symptoms_dict.keys(), n=1, cutoff=0.6)
            if closest_matches:
                closest = closest_matches[0]
                print(f"[Approximated] '{symptom}' → '{closest}'")
                input_vector[symptoms_dict[closest]] = 1
            else:
                print(f"[Skipped] '{symptom}' not recognized and no close match found.")
    
    prediction = svc.predict([input_vector])[0]
    print(prediction)
    return diseases_list[prediction]

def RecommendHospitalsOnDiseases(disease):
    # Filter the hospitals based on the disease
    filtered_hospitals = ethiopian_hospitals[ethiopian_hospitals['Disease'].str.contains(disease, case=False, na=False)]
    # Convert the filtered DataFrame to a list of dictionaries
    hospital_list = filtered_hospitals.to_dict(orient='records')
    return hospital_list

def RecommendHospitalsOnRegions(region):
    # Filter the hospitals based on the region
    filtered_hospitals = ethiopian_hospitals[ethiopian_hospitals['Region'].str.contains(region, case=False, na=False)]
    # Convert the filtered DataFrame to a list of dictionaries
    hospital_list = filtered_hospitals.to_dict(orient='records')
    return hospital_list
def RecommendHospitalsOnCities(city):
        # Filter the hospitals based on the region
    filtered_hospitals = ethiopian_hospitals[ethiopian_hospitals['City'].str.contains(city, case=False, na=False)]
    # Convert the filtered DataFrame to a list of dictionaries
    hospital_list = filtered_hospitals.to_dict(orient='records')
    return hospital_list
def RecommendHospitalsOnServices(service):
    # Filter the hospitals based on the service
    filtered_hospitals = ethiopian_hospitals[ethiopian_hospitals['Services'].str.contains(service, case=False, na=False)]
    # Convert the filtered DataFrame to a list of dictionaries
    hospital_list = filtered_hospitals.to_dict(orient='records')
    return hospital_list
def RecommendHospitalsOnSpecialties(specialty):
    # Filter the hospitals based on the specialty
    filtered_hospitals = ethiopian_hospitals[ethiopian_hospitals['Specialties'].str.contains(specialty, case=False, na=False)]
    # Convert the filtered DataFrame to a list of dictionaries
    hospital_list = filtered_hospitals.to_dict(orient='records')
    return hospital_list


# # print(hospital_list)
# hospitals_details = []
# for hospital in hospital_list:
#     hospital_details = {
#         'Hospital_Name': hospital['Hospital_Name'],
#         'Location': f"{hospital['Region']}, {hospital['City']}",
#     }
#     hospitals_details.append(hospital_details)



# creating routes========================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_symptoms")
def get_symptoms():
    df = pd.read_csv("symtoms_df.csv")

    # Get all symptoms from Symptom_1 to Symptom_4 columns, flatten and clean
    symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
    all_symptoms = set()

    for col in symptom_cols:
        all_symptoms.update(df[col].dropna().str.strip().str.lower())

    return jsonify(sorted(all_symptoms))

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        # if symptoms not in symptoms_dict.keys():
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(' ')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            print(user_symptoms)
            predicted_disease = get_predicted_value(user_symptoms)
            hospitals = []
            for symptom in user_symptoms:
                if symptom in symptoms_dict:
                    hospital_recommendations = RecommendHospitalsOnDiseases(predicted_disease)
                    hospitals.append(hospital_recommendations)
                if symptom in regions_dict:
                    hospital_recommendations = RecommendHospitalsOnRegions(symptom)
                    hospitals.append(hospital_recommendations)
                if symptom in cities_list:
                    hospital_recommendations = RecommendHospitalsOnCities(symptom)
                    hospitals.append(hospital_recommendations)
                if symptom in Services_list:
                    hospital_recommendations = RecommendHospitalsOnServices(symptom)
                    hospitals.append(hospital_recommendations)
                if symptom in Specialties_list:
                    hospital_recommendations = RecommendHospitalsOnSpecialties(symptom)
                    hospitals.append(hospital_recommendations)
            # hospital_recommendations = list(set(hospitals))
            if hospitals:
                hospital_recommendations = hospitals[0]
            else:
                hospital_recommendations = hospitals
            print(hospital_recommendations)
            # for symptom in user_symptoms:
            #     if symptom in symptoms_dict.keys():
            #         hospital_recommendations = RecommendHospitalsOnDiseases(symptom)
            # hospital_recommendations = RecommendHospitalsOnDiseases(predicted_disease)
    
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            medications_list = ast.literal_eval(medications[0])
            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet,
                                   workout=workout, hospitals=hospital_recommendations)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)