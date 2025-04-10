from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model from the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define unique values for categorical columns based on value_counts
unique_categories = {
    'preferred_study_level': ['Postgraduate', 'Undergraduate', 'Vocational', 'Other', 'Doctorate'],
    'phd_supervisor': ['No', 'Yes'],
    'supervisor_approval': ['No', 'Yes'],
    'intended_study_area': ['Business & Economics', 'Engineering', 'Other', 'Computer Science and Information Technology', 'Sciences'] + [f'Area {i}' for i in range(265)],  # Placeholder
    'preferred_dest_1': ['Other', 'Canada', 'UK', 'Australia', 'USA', 'Ireland', 'New Zealand'],
    'preferred_dest_2': ['Other', 'Canada', 'UK', 'Australia', 'USA', 'Ireland', 'New Zealand'],
    'preferred_dest_3': ['Other', 'Canada', 'Australia', 'UK', 'USA', 'New Zealand', 'Ireland'],
    'study_plan': ['01-2021', '09-2021'] + [f'Date {i}' for i in range(5)],  # Placeholder
    'uni_contacted': ['No', 'Yes'],
    'application_process': ['No', 'Yes'],
    'current_status': ['Student', 'Working professional'],
    'work_experience': ['0', '1-3', '4-6', '7+', '<1'],
    'passport': ['Yes', 'No'],
    'academic_backlogs': ['No', 'Yes'],
    'number_of_backlogs': ['0', '<5', '6', '>10'],
    'budget': ['Not provided', '10-20', '<10', '20-30', '>30'],
    'funding_source': ['Parents asset', 'Combined self /parent funding and bank loan', 'Bank loan', 'Self funding', 'Other', 'Not provided', 'Sponsorship by relatives'],
    'parents_occupation': ['In-Service', 'Own business', 'Not provided'],
    'english_test_taken': ['No', 'Yes'],
    'english_test_name': ['Not Taken', 'IELTS', 'Other', 'GRE', 'TOEFL', 'GMAT'],
    'family_abroad': ['Yes', 'No'],
    'other_consultants': ['No', 'Not Provided', 'No did not contact', 'yes', 'not remember']
}

# Create LabelEncoders for each categorical column using unique categories
label_encoders = {}
for column, categories in unique_categories.items():
    le = LabelEncoder()
    le.fit(categories)
    label_encoders[column] = le

# Define route for the form
@app.route('/', methods=['GET', 'POST'])
def form():
    prediction = None  # Initialize prediction to None
    if request.method == 'POST':
        # Get form data
        input_data = {
            'preferred_study_level': request.form['preferred_study_level'],
            'phd_supervisor': request.form['phd_supervisor'],
            'supervisor_approval': request.form['supervisor_approval'],
            'intended_study_area': request.form['intended_study_area'],
            'preferred_dest_1': request.form['preferred_dest_1'],
            'preferred_dest_2': request.form['preferred_dest_2'],
            'preferred_dest_3': request.form['preferred_dest_3'],
            'study_plan': request.form['study_plan'],
            'uni_contacted': request.form['uni_contacted'],
            'application_process': request.form['application_process'],
            'current_status': request.form['current_status'],
            'work_experience': request.form['work_experience'],
            'birth_year': int(request.form['birth_year']),
            'passport': request.form['passport'],
            'academic_backlogs': request.form['academic_backlogs'],
            'number_of_backlogs': request.form['number_of_backlogs'],
            'budget': request.form['budget'],
            'funding_source': request.form['funding_source'],
            'parents_occupation': request.form['parents_occupation'],
            'english_test_taken': request.form['english_test_taken'],
            'english_test_name': request.form['english_test_name'],
            'english_test_score': float(request.form['english_test_score']),
            'family_abroad': request.form['family_abroad'],
            'other_consultants': request.form['other_consultants'],
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables using the LabelEncoders
        for column in input_df.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                input_df[column] = label_encoders[column].transform(input_df[column])

        # Predict the outcome
        prediction = model.predict(input_df)[0]

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
