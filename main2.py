import streamlit as st
import pandas as pd
import joblib

# Load the trained model
best_model = joblib.load('heart.pkl')

# Function to preprocess user input
def preprocess_input(trestbps, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal):
    input_data = pd.DataFrame({
        'trestbps': [trestbps],
        'oldpeak': [oldpeak],
        'sex_1': [1 if sex == 'male' else 0],
        'cp_1': [1 if cp == 'atypical angina' else 0],
        'cp_2': [1 if cp == 'typical angina' else 0],
        'cp_3': [1 if cp == 'non-anginal' else 0],
        'fbs_1': [1 if fbs > 120 else 0],
        'restecg_1': [1 if restecg == 'normal' else 0],
        'restecg_2': [1 if restecg == 'ST-T wave abnormality' else 0],
        'exang_1': [1 if exang == 'yes' else 0],
        'slope_1': [1 if slope == 'upsloping' else 0],
        'ca_1': [1 if ca == 0 else 0],
        'ca_2': [1 if ca == 1 else 0],
        'ca_3': [1 if ca == 2 else 0],
        'ca_4': [1 if ca == 3 else 0],
        'thal_1': [1 if thal == 'error' else 0],
        'thal_3': [1 if thal == 'reversible defect' else 0]
    })
    return input_data

# Function to predict heart disease
def predict_heart_disease(input_data):
    prediction = best_model.predict(input_data)
    probability = best_model.predict_proba(input_data)[0][1]
    return prediction, probability

# Main function to run the Streamlit app
def main():
    st.title('Heart Disease Prediction')

    # Get user input
    trestbps = st.number_input('Enter resting Blood Pressure:')
    oldpeak = st.number_input('Enter ST depression induced by exercise relative to rest (decimal):')
    sex = st.selectbox('Select gender:', ['male', 'female'])
    cp = st.selectbox('Select type of Chest Pain:', ['atypical angina', 'typical angina', 'non-anginal', 'asymptomatic'])
    fbs = st.number_input('Enter Fasting Blood Sugar:')
    restecg = st.selectbox('Select type of Rest ECG abnormalities:', ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
    exang = st.selectbox('Is it exercise induced angina:', ['yes', 'no'])
    slope = st.selectbox('Select type of slope of ST/heart rate:', ['upsloping', 'flat', 'downsloping'])
    ca = st.number_input('Enter number of colored cells: (0, 1, 2, 3):', min_value=0, max_value=3, step=1)
    thal = st.selectbox('Enter Thalassemia classification:', ['error', 'fixed defect', 'normal', 'reversible defect'])

    # Submit button
    if st.button('Predict'):
        # Preprocess input data
        input_data = preprocess_input(trestbps, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal)
        
        # Predict heart disease
        prediction, probability = predict_heart_disease(input_data)
        
        # Display prediction and probability
        if prediction[0] == 1:
            st.write("YES, YOU HAVE HEART DISEASE")
        else:
            st.write("NO, YOU DON'T HAVE HEART DISEASE")
        st.write(f'Probability: {probability*100:.2f}%')

if __name__ == "__main__":
    main()
