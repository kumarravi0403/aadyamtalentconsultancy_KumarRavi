import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Phase 1: Data Loading & Preprocessing
# ==========================================        

# 1. LOAD YOUR DATASET
# Replace 'student_data.csv' with the actual name of your 6k dataset file
df = pd.read_csv('cleaned_data.csv')
#1.1 Thresholding: Create the target variable
# I am setting 70 as passing marks
passing_marks = 70
df['Pass_Fail'] = (df['Exam_Score']>=passing_marks).astype(int)

# (Optional but recommended) Print the first 5 rows to verify it loaded correctly
print("--- Dataset Preview ---")
print(df.head(), "\n")

# 2. HANDLE MISSING VALUES (Crucial for real datasets)
# If your dataset has blank cells, this removes those rows to prevent errors
df = df.dropna()

# 3. DEFINE FEATURES (X) AND TARGET (y)
# IMPORTANT: Change these string names to exactly match the headers in your CSV!
feature_columns = ['Hours_Studied', 'Attendance', 'Previous_Scores'] 

X = df[feature_columns]
y = df['Pass_Fail']

# ==========================================
# Phase 2: Splitting & Scaling
# ==========================================

# 4. TRAIN-TEST SPLIT (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. FEATURE SCALING (Standardization)
# We scale the data so large numbers (like scores out of 100) don't overpower 
# small numbers (like study hours out of 10).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Phase 3: Model Training
# ==========================================

# 6. INITIALIZE AND TRAIN THE LOGISTIC REGRESSION MODEL
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ==========================================
# Phase 4: Model Evaluation
# ==========================================

# 7. MAKE PREDICTIONS ON THE UNSEEN TEST DATA
y_pred = model.predict(X_test_scaled)

# 8. GENERATE DELIVERABLES
accuracy = accuracy_score(y_test, y_pred)

print("--- Final Model Evaluation Deliverables ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
# This shows: [[True Negatives, False Positives], 
#              [False Negatives, True Positives]]
print(confusion_matrix(y_test, y_pred))
print("\n")

print("Detailed Classification Report:")
# This provides Precision, Recall, and F1-Score for deep evaluation
print(classification_report(y_test, y_pred))