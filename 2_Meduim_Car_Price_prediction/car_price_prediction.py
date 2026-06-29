import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the Dataset
# Make sure your CSV file is named 'car_dataset.csv' and is in the same directory
df = pd.read_csv('car_price_dataset.csv')

# Define which columns are numeric and which are categorical based on your CSV
numeric_features = ['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']

# Convert numeric columns to numeric, coercing errors to NaN (handles empty strings or spaces)
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with missing values to ensure smooth training
df = df.dropna()

# 2. Data Preprocessing
# Drop the 'name' column as high-cardinality text data requires advanced NLP to process effectively
X = df.drop(['selling_price', 'name'], axis=1)
y = df['selling_price']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create transformers for scaling numbers and encoding text
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine them into a single preprocessor step
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Model Training
# Initialize the Random Forest pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Training the model... this might take a few seconds.")
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): Rs. {mae:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f} (Closer to 1.0 is better)")

# 5. Save the Model (Deliverable Requirement)
joblib.dump(model, 'car_price_predictor.pkl')
print("\nModel saved as 'car_price_predictor.pkl'")

# 6. Data Visualization (Deliverable Requirement)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='darkorange')

# Draw the line of perfect prediction
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) 

plt.xlabel('Actual Selling Price (Rs.)')
plt.ylabel('Predicted Selling Price (Rs.)')
plt.title('Actual vs Predicted Car Prices')
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file for your GitHub repository
plt.savefig('actual_vs_predicted.png')
print("Visualization saved as 'actual_vs_predicted.png'")

# Display the plot
# plt.show()
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
# plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["selling_price"], bins=20)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("price_distribution.png")
# plt.show()

# Extract feature importances from the pipeline
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = model.named_steps['regressor'].feature_importances_
importance = pd.Series(importances, index=feature_names)

importance.sort_values().tail(15).plot(kind="barh", figsize=(8,5))
plt.title("Feature Importance (Top 15)")
plt.tight_layout()
plt.savefig("feature_importance.png")
# plt.show()

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse:.2f}")