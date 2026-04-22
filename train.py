"""
Generate synthetic student dataset and train a Random Forest model.
Features: study_hours, sleep_hours, past_marks, attendance_pct, extra_activities
Target: final_marks
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

np.random.seed(42)
N = 2000

# Simulate realistic student data
study_hours = np.random.normal(5, 2, N).clip(0, 16)
sleep_hours = np.random.normal(7, 1.5, N).clip(3, 12)
past_marks = np.random.normal(65, 15, N).clip(20, 100)
attendance_pct = np.random.normal(78, 15, N).clip(30, 100)
extra_activities = np.random.randint(0, 6, N).astype(float)  # 0-5 activities

# Final marks formula (realistic & learnable)
# Each feature contributes a meaningful, independent signal
study_score   = study_hours / 16.0 * 35          # up to 35 points
sleep_score   = np.where(
    sleep_hours < 6, (sleep_hours - 3) * 3,       # penalty for under-sleep
    np.where(sleep_hours > 9, (12 - sleep_hours) * 2, 15)  # penalty for over-sleep
)                                                  # up to 15 points
past_score    = past_marks * 0.30                 # up to 30 points (past strongly predicts future)
attend_score  = attendance_pct / 100.0 * 12       # up to 12 points
activity_score = extra_activities * 1.2           # up to 6 points (slight boost)
noise         = np.random.normal(0, 3, N)         # small Gaussian noise

final_marks = study_score + sleep_score + past_score + attend_score + activity_score + noise
final_marks = final_marks.clip(10, 98)

df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'past_marks': past_marks,
    'attendance_pct': attendance_pct,
    'extra_activities': extra_activities,
    'final_marks': final_marks
})

df = df.round(2)
df.to_csv('student_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.describe().round(2))

# Prepare features
X = df.drop('final_marks', axis=1)
y = df['final_marks']

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models and pick best
models = {
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

best_model = None
best_r2 = -1
best_name = ''

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{name}: R2={r2:.4f}, MAE={mae:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

# Save model artifacts
joblib.dump(best_model, 'student_model.joblib')
joblib.dump(scaler, 'student_scaler.joblib')
joblib.dump(feature_names, 'student_features.joblib')

print("Saved: student_model.joblib, student_scaler.joblib, student_features.joblib")
print("Training complete!")
