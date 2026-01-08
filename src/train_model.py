import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import numpy as np

def train_and_evaluate(data_path, models_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_pickle(data_path)
    
    # Split data
    X = df['text']
    y_class = df['problem_class']
    y_score = df['problem_score']
    
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.1, random_state=42 # Tried increasing training data to 90%
    )
    
    # --- Classification Model (Optimized V5: Fine-tuned Ensemble) ---
    print("\nTraining Classification Model (Voting: SVC + LR + RF + NB)...")
    
    # Estimators
    svc = LinearSVC(random_state=42, dual='auto', class_weight='balanced', C=1.0)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=1.0)
    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight='balanced')
    nb = MultinomialNB(alpha=0.05) # Lower alpha for less smoothing
    
    voting_clf = VotingClassifier(estimators=[('svc', svc), ('lr', lr), ('rf', rf), ('nb', nb)], voting='hard')
    
    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
        ('scaler', MaxAbsScaler()),
        ('classifier', voting_clf)
    ])
    
    clf_pipeline.fit(X_train, y_class_train)
    
    print("Evaluating Classification Model...")
    y_class_pred = clf_pipeline.predict(X_test)
    report = classification_report(y_class_test, y_class_pred, output_dict=True)
    accuracy = report['accuracy']
    print(classification_report(y_class_test, y_class_pred))
    print(f"!! CLASSIFICATION ACCURACY: {accuracy:.4f} !!")
    
    # --- Regression Model ---
    print("\nTraining Regression Model...")
    reg_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    reg_pipeline.fit(X_train, y_score_train)
    
    print("Evaluating Regression Model...")
    y_score_pred = reg_pipeline.predict(X_test)
    mae = mean_absolute_error(y_score_test, y_score_pred)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # --- Save Models ---
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print(f"\nSaving models to {models_dir}...")
    joblib.dump(clf_pipeline, os.path.join(models_dir, 'classifier_model.pkl'))
    joblib.dump(reg_pipeline, os.path.join(models_dir, 'regressor_model.pkl'))
    print("Models saved.")

if __name__ == "__main__":
    data_path = os.path.join("data", "processed_data.pkl")
    models_dir = "models"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run preprocessing.py first.")
        exit(1)
        
    train_and_evaluate(data_path, models_dir)
