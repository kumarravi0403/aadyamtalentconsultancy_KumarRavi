import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(
    "StudentPerformanceFactors.csv"
)

print("Original Shape:", df.shape)

# Select useful columns
df = df[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Exam_Score"
    ]
]

# Convert text columns to numbers
encoder = LabelEncoder()

# categorical_columns = [
#     "City",
#     "Furnishing Status",
#     "Tenant Preferred",
#     "Area Type",
# ]

# for col in categorical_columns:
#     df[col] = encoder.fit_transform(df[col])

print("\nProcessed Dataset:")
print(df.head())

# Save cleaned dataset
output_path = "cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)