import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Ensure data/models exist
if not os.path.exists('data/processed_data.pkl'):
    print("Data not found!")
    exit(1)

# Load data
print("Loading data...")
df = pd.read_pickle('data/processed_data.pkl')
X = df['text']
y_class = df['problem_class']
y_score = df['problem_score']

# Split same as training
print("Splitting data...")
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.1, random_state=42
)

# Load Models
print("Loading models...")
clf = joblib.load('models/classifier_model.pkl')
reg = joblib.load('models/regressor_model.pkl')

# Predict
print("Predicting...")
y_class_pred = clf.predict(X_test)
y_score_pred = reg.predict(X_test)

# Metrics
acc = accuracy_score(y_class_test, y_class_pred)
mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))

print(f"Accuracy: {acc}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Save Metrics to text file for report reading
with open('report_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

# Confusion Matrix
print("Generating Confusion Matrix...")
labels = ['easy', 'medium', 'hard']
cm = confusion_matrix(y_class_test, y_class_pred, labels=labels)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Easy', 'Medium', 'Hard'], yticklabels=['Easy', 'Medium', 'Hard'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Ensemble Model')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# Class Distribution Plot
print("Generating Class Distribution...")
plt.figure(figsize=(8, 5))
df['problem_class'].value_counts().plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'])
plt.title('Problem Difficulty Distribution')
plt.xlabel('Difficulty Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Saved class_distribution.png")
