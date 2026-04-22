from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'student_model.joblib'
SCALER_PATH = 'student_scaler.joblib'
FEATURES_PATH = 'student_features.joblib'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model loaded successfully.")
else:
    model, scaler, feature_names = None, None, None
    print("WARNING: Model not found. Run train.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 500
    try:
        data = request.json

        study_hours = float(data.get('study_hours', 0))
        sleep_hours = float(data.get('sleep_hours', 7))
        past_marks = float(data.get('past_marks', 50))
        attendance_pct = float(data.get('attendance_pct', 75))
        extra_activities = int(data.get('extra_activities', 0))

        # Validation
        errors = []
        if not (0 <= study_hours <= 16): errors.append("Study hours must be 0-16.")
        if not (3 <= sleep_hours <= 12): errors.append("Sleep hours must be 3-12.")
        if not (0 <= past_marks <= 100): errors.append("Past marks must be 0-100.")
        if not (0 <= attendance_pct <= 100): errors.append("Attendance must be 0-100%.")
        if not (0 <= extra_activities <= 5): errors.append("Extra activities must be 0-5.")
        if errors:
            return jsonify({'error': ' '.join(errors)}), 400

        df_input = pd.DataFrame([{
            'study_hours': study_hours,
            'sleep_hours': sleep_hours,
            'past_marks': past_marks,
            'attendance_pct': attendance_pct,
            'extra_activities': extra_activities
        }])

        scaled = scaler.transform(df_input)
        prediction = model.predict(scaled)[0]
        predicted_marks = round(max(10, min(100, prediction)), 1)

        # Grade label
        if predicted_marks >= 90:
            grade, grade_label = 'A+', 'Outstanding'
        elif predicted_marks >= 80:
            grade, grade_label = 'A', 'Excellent'
        elif predicted_marks >= 70:
            grade, grade_label = 'B', 'Good'
        elif predicted_marks >= 60:
            grade, grade_label = 'C', 'Average'
        elif predicted_marks >= 50:
            grade, grade_label = 'D', 'Below Average'
        else:
            grade, grade_label = 'F', 'Needs Improvement'

        return jsonify({
            'predicted_marks': predicted_marks,
            'grade': grade,
            'grade_label': grade_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
