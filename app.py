import streamlit as st
import pickle
import numpy as np

# Load the scaler, model, and class names
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score, average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict using the model
    probabilities = model.predict_proba(scaled_features)
    
    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs


# Streamlit application interface

st.image('images/img.png')

st.title('Career Path Recommendation')

# Sidebar with two columns for inputs
st.sidebar.header('Input Features')

col1, col2 = st.sidebar.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    part_time_job = st.checkbox("Part-time Job")
    absence_days = st.number_input("Number of Absence Days", min_value=0, max_value=365, value=0)
    extracurricular_activities = st.checkbox("Extracurricular Activities")
    weekly_self_study_hours = st.slider("Weekly Self Study Hours", min_value=0, max_value=50, value=10)
    math_score = st.slider("Math Score", min_value=0, max_value=100, value=75)
    history_score = st.slider("History Score", min_value=0, max_value=100, value=75)

with col2:
    physics_score = st.slider("Physics Score", min_value=0, max_value=100, value=75)
    chemistry_score = st.slider("Chemistry Score", min_value=0, max_value=100, value=75)
    biology_score = st.slider("Biology Score", min_value=0, max_value=100, value=75)
    english_score = st.slider("English Score", min_value=0, max_value=100, value=75)
    geography_score = st.slider("Geography Score", min_value=0, max_value=100, value=75)
    total_score = st.number_input("Total Score", min_value=0, max_value=1500, value=1000)
    average_score = st.number_input("Average Score", min_value=0.0, max_value=100.0, value=75.0)

# Display the input values in two columns
st.subheader("Input Values")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Gender: {gender}")
    st.write(f"Part-time Job: {part_time_job}")
    st.write(f"Absence Days: {absence_days}")
    st.write(f"Extracurricular Activities: {extracurricular_activities}")
    st.write(f"Weekly Self Study Hours: {weekly_self_study_hours}")
    st.write(f"Math Score: {math_score}")
    st.write(f"History Score: {history_score}")
    st.write(f"Physics Score: {physics_score}")

with col2:
    st.write(f"Chemistry Score: {chemistry_score}")
    st.write(f"Biology Score: {biology_score}")
    st.write(f"English Score: {english_score}")
    st.write(f"Geography Score: {geography_score}")
    st.write(f"Total Score: {total_score}")
    st.write(f"Average Score: {average_score}")

# Call the Recommendations function when the button is pressed
if st.button("Get Career Recommendations"):
    recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                      weekly_self_study_hours, math_score, history_score, physics_score,
                                      chemistry_score, biology_score, english_score, geography_score,
                                      total_score, average_score)

    st.subheader("Top Recommended Careers with Probabilities")
    for class_name, probability in recommendations:
        st.write(f"{class_name}: {probability:.2f}")
