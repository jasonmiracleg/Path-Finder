from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

with open('../stacked_balanced_weigthed.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    skills = [
        'Coding Skills', 
        'Communication Skills', 
        'Problem Solving Skills', 
        'Teamwork Skills', 
        'Analytical Skills', 
        'Presentation Skills', 
        'Networking Skills'
    ]
    return render_template("index.html", skills=skills)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    soft_skills = [
        int(data['communication_skills']),
        int(data['teamwork_skills']),
        int(data['presentation_skills']),
        int(data['networking_skills']),
        int(data['problem_solving_skills'])
    ]
    
    technical_skills = [
        int(data['coding_skills']),
        int(data['field_specific_courses']),
        int(data['research_experience']),
        int(data['industry_certifications']),
        int(data['analytical_skills'])
    ]
    
    proactive_engagement = [
        int(data['leadership_positions']),
        int(data['extracurricular_activities']),
        int(data['projects']),
        int(data['internships']),
    ]
    
    avg_soft_skills = sum(soft_skills) / len(soft_skills)
    avg_technical_skills = sum(technical_skills) / len(technical_skills)
    avg_proactive_engagement = sum(proactive_engagement) / len(proactive_engagement)
    
    # Extract features from the incoming data
    features = [
        float(data['gpa']),
        avg_soft_skills,
        avg_technical_skills,
        avg_proactive_engagement
    ]
    
    input_data = pd.DataFrame([features], columns=['GPA', 'Soft_Skills', 'Technical_Skills', 'Proactive_Engagement'])
    prediction = model.predict(input_data)

    predicted_field = int(prediction[0])
    # Return the prediction as a JSON response
    return jsonify({'field': predicted_field})

if __name__ == '__main__':
    app.run(port=5000, debug=True)