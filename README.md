# AutoJudge: Programming Problem Difficulty Predictor
AutoJudge is an intelligent system that predicts the difficulty class (Easy, Medium, Hard) and a numerical difficulty score for programming problems based solely on their textual description.

## DRIVE link for report -
https://drive.google.com/file/d/1Jge6gGHWQ_eZpDHCMrnS4WSR7gpC1u5h/view?usp=sharing

## DRIVE link for demo video -
https://drive.google.com/file/d/13fcWimF-45w5i6a5tH3LqwY3qMAxYaMf/view?usp=sharing

## Project Overview

This project uses Machine Learning to analyze the text of programming problems. It utilizes a **Random Forest Classifier** for categorizing difficulty and a **Random Forest Regressor** for predicting the difficulty score. The system is wrapped in a simple **Streamlit** web interface for easy usage.

## Dataset
The project uses `problems_data.jsonl`, which contains:
- `title`, `description`, `input_description`, `output_description`
- `problem_class` (Target: Easy/Medium/Hard)
- `problem_score` (Target: Numerical score)

## Approach
1.  **Data Preprocessing**:
    -   Missing values in text fields are handled.
    -   `description`, `input_description`, and `output_description` are combined into a single text feature.
2.  **Feature Extraction**:
    -   **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert text into numerical vectors, capturing the importance of words like "graph", "tree", "dp", etc.
3.  **Modeling**:
    -   **Classification**: Random Forest Classifier.
    -   **Regression**: Random Forest Regressor.
    -   **Evaluation**: Accuracy for classification; MAE/RMSE for regression.

## Steps to Run Locally

1.  **Clone the repository** (if applicable).
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Preprocess Data**:
    ```bash
    python src/preprocessing.py
    ```
    This creates `data/processed_data.pkl`.
4.  **Train Models**:
    ```bash
    python src/train_model.py
    ```
    This trains the models and saves them to `models/`.
5.  **Run Web Interface**:
    ```bash
    streamlit run src/app.py
    ```
    The app will open in your browser.

## Web Interface
The web UI allows you to:
-   Paste the Problem Description, Input Description, and Output Description.
-   Click "Predict Difficulty".
-   View the predicted Class (Easy/Medium/Hard) and Score.

## Evaluation Metrics
(These will be populated after running `train_model.py` - check the console output)
-   **Classification Accuracy**: 55.10%
-   **Regression RMSE**: 2.0643

