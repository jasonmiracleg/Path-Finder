from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

with open('../PathFinder.pkl', 'rb') as file:
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

    skills = [
        int(data['communication_skills']),
        int(data['teamwork_skills']),
        int(data['presentation_skills']),
        int(data['networking_skills']),
        int(data['problem_solving_skills']),
        int(data['analytical_skills']),
        int(data['coding_skills'])
    ]
    
    technical_skills = [
        int(data['field_specific_courses']),
        int(data['research_experience']),
        int(data['industry_certifications']),
    ]
    
    proactive_engagement = [
        int(data['leadership_positions']),
        int(data['extracurricular_activities']),
        int(data['projects']),
        int(data['internships']),
    ]
    
    avg_skills = sum(skills) / len(skills)
    avg_technical_skills = sum(technical_skills) / len(technical_skills)
    avg_proactive_engagement = sum(proactive_engagement) / len(proactive_engagement)
    
    # Extract features from the incoming data
    features = [
        float(data['gpa']),
        avg_technical_skills,
        avg_proactive_engagement,
        avg_skills
    ]
    
    input_data = pd.DataFrame([features], columns=['GPA', 'Technical_Skills', 'Proactive_Engagement', 'Skills_Average'])
    prediction = model.predict(input_data)

    predicted_field = int(prediction[0])
    # Return the prediction as a JSON response
    return render_template("result.html", result=predicted_field)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
