from flask import Flask, request, render_template, jsonify 
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

sym_des = pd.read_csv("datasets/symptoms.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
doctor = pd.read_csv("datasets/doctor.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

svc = pickle.load(open('models/svc.pkl','rb'))

def helper(dis): 
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    doc = doctor[doctor['disease'] == dis] ['doctor']

    return med, doc

symptoms_dict = {'itching':  0,
'red_eyes': 1,
'skin_redness': 2,
'hives': 3,
'sneezing': 4,
'tremors': 5,
'feverishness': 6,
'stiffness': 7,
'abdominal_discomfort': 8,
'heartburn': 9,
'mouth_sores': 10,
'muscle_twitching': 11,
'nausea_and_diarrhea': 12,
'urinary_burning': 13,
'frequent_urination': 14,
'drowsiness': 15,
'sudden_weight_gain': 16,
'nervousness': 17,
'numbness_in_limbs': 18,
'mood_swings': 19,
'appetite_loss': 20,
'agitation': 21,
'fatigue': 22,
'throat_pain': 23,
'blood_sugar_fluctuations': 24,
'coughing': 25,
'high_temperature': 26,
'sunken_eyes': 27,
'shortness_of_breath': 28,
'profuse_sweating': 29,
'thirst': 30,
'stomach_upset': 31,
'throbbing_headache': 32,
'yellowish_skin_tinge': 33,
'dark_urine': 34,
'eye_pain': 35,
'backache': 36,
'bowel_irregularity': 37,
'stomach_cramps': 38,
'bowel_discomfort': 39,
'slight_fever': 40,
'yellowish_urine': 41,
'yellowing_of_skin': 42,
'liver_failure': 43,
'fluid_retention': 44,
'abdominal_swelling': 45,
'swollen_lymph_nodes': 46,
'general_discomfort': 47,
'vision_blurring': 48,
'excess_mucus': 49,
'throat_inflammation': 50,
'eye_redness': 51,
'sinus_pressure': 52,
'nasal_drip': 53,
'congestion': 54,
'chest_discomfort': 55,
'limb_weakness': 56,
'rapid_pulse': 57,
'bowel_pain': 58,
'anal_discomfort': 59,
'bloody_stool': 60,
'anal_itching': 61,
'neck_soreness': 62,
'lightheadedness': 63,
'muscle_cramps': 64,
'skin_bruising': 65,
'weight_problem': 66,
'leg_swelling': 67,
'vein_swelling': 68,
'facial_swelling': 69,
'thyroid_enlargement': 70,
'brittle_nails': 71,
'swollen_extremities': 72,
'insatiable_hunger': 73,
'illicit_sexual_activity': 74,
'lip_dryness': 75,
'slurred_speech': 76,
'knee_pain': 77,
'hip_pain': 78,
'muscle_weakness': 79,
'stiff_neck': 80,
'joint_swelling': 81,
'joint_stiffness': 82,
'dizziness': 83,
'loss_of_balance': 84,
'unsteadiness': 85,
'weakness_on_one_side': 86,
'loss_of_smell': 87,
'bladder_pain': 88,
'foul_urine_smell': 89,
'constant_urination_sensation': 90,
'gas_passing': 91,
'internal_itchiness': 92,
'toxic_look': 93,
'mental_distress': 94,
'anger': 95,
'muscle_discomfort': 96,
'consciousness_change': 97,
'body_rash': 98,
'abdominal_discomfort': 99,
'menstrual_irregularity': 100,
'skin_discoloration': 101,
'teary_eyes': 102,
'increased_hunger': 103,
'excessive_urination': 104,
'family_health_history': 105,
'thick_mucus': 106,
'rust-colored_sputum': 107,
'concentration_loss': 108,
'visual_impairment': 109,
'blood_transfusion_history': 110,
'unsanitary_injections_history': 111,
'coma': 112,
'abdominal_bleeding': 113,
'abdominal_swelling': 114,
'alcohol_abuse_history': 115,
'fluid_retention': 116,
'bloody_sputum': 117,
'prominent_calf_veins': 118,
'heart_palpitations': 119,
'walking_pain': 120,
'pus_filled_bumps': 121,
'blackheads': 122,
'scaly_skin': 123,
'skin_peeling': 124,
'silver_dust_like_scales': 125,
'nail_pitting': 126,
'inflammatory_nails': 127,
'skin_blisters': 128,
'nose_soreness': 129,
'hirsutism': 130,
'informed_guess_for_diagnosis': 131
}

diseases_list = {0:  '(vertigo) Paroymsal Positional Vertigo',
1:  'AIDS',
2:  'Acne',
3:  'Alcoholic hepatitis',
4:  'Allergy',
5:  'Arthritis',
6:  'Bronchial Asthma',
7:  'Cervical spondylosis',
8:  'Chicken pox',
9:  'Chronic cholestasis',
10:  'Common Cold',
11:  'Dengue',
12:  'Diabetes ',
13:  'Dimorphic hemmorhoids(piles)',
14:  'Drug Reaction',
15:  'Fungal infection',
16:  'GERD',
17:  'Gastroenteritis',
18:  'Heart attack',
19:  'Hepatitis B',
20:  'Hepatitis C',
21:  'Hepatitis D',
22:  'Hepatitis E',
23:  'Hypertension ',
24:  'Hyperthyroidism',
25:  'Hypoglycemia',
26:  'Hypothyroidism',
27:  'Impetigo',
28:  'Jaundice',
29:  'Malaria',
30:  'Migraine',
31:  'Osteoarthristis',
32:  'Paralysis (brain hemorrhage)',
33:  'Peptic ulcer diseae',
34:  'Pneumonia',
35:  'Psoriasis',
36:  'Tuberculosis',
37:  'Typhoid',
38:  'Urinary tract infection',
39:  'Varicose veins',
40:  'hepatitis A'
}

def get_predicted_value(patient_symptoms): 
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms: 
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

@app.route("/")
def index(): 
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home(): 
    if request.method == 'POST': 
        symptoms = request.form.get('symptoms')

        print(symptoms)
        if symptoms =="Symptoms": 
            message = "There is some error"
            return render_template('index.html', message=message)
        else: 
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            informed_guess_for_diagnosis = get_predicted_value(user_symptoms)
            precautions, medications, doctor = helper(informed_guess_for_diagnosis)

            my_precautions = []
            for i in precautions[0]: 
                my_precautions.append(i)

            return render_template('index.html', informed_guess_for_diagnosis = informed_guess_for_diagnosis, medications=medications, doctor=doctor)

    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug=True)