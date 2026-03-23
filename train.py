"""
Iris Species Classification Model Training
Student ID: st126112
Algorithm: Random Forest Classifier
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)
print("Training completed!")

# Evaluate
print("\n" + "="*50)
print("MODEL EVALUATION - Student ID: st126112")
print("="*50)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)
print("="*50)

# Save model
model_filename = 'model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved successfully to {model_filename}")
print(f"Model type: {type(model).__name__}")
print(f"Number of estimators: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
