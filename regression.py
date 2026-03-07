import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_excel("data/detailed_food_shelf_life_dataset.xlsx")

# Encode categorical columns
le = LabelEncoder()

for col in ['Food','Dish_Type','Storage_Type','Humidity_Level',
            'Container_Type','Reheated','CNN_Result','Safe_to_Eat']:
    data[col] = le.fit_transform(data[col])

# Target column
y = data['Safe_to_Eat']

# Features
X = data.drop(['Safe_to_Eat','Risk_Level'], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "shelf_life_model.pkl")

print("Regression model saved as shelf_life_model.pkl")