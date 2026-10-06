# Liver Disease Prediction System

## Project Overview
This project is a Machine Learning based Liver Disease Prediction System. It uses an ensemble method combining Gradient Boosting and AdaBoost classifiers to predict the presence of liver disease. The system includes proper data preprocessing, feature selection, k-fold cross-validation, and is deployed via a Streamlit web application with a simple, user-friendly interface.

## Dataset
The dataset used is the **UCI Indian Liver Patient Dataset (ILPD)**. It contains records of liver patient and non-liver patient data.
The target has been converted to:
- 1 = Liver Disease
- 0 = No Liver Disease

Gender has been encoded as:
- 1 = Male
- 0 = Female

## Features
The model utilizes the following 10 features:
1. Age
2. Gender
3. Total Bilirubin
4. Direct Bilirubin
5. Alkaline Phosphotase
6. Alamine Aminotransferase
7. Aspartate Aminotransferase
8. Total Proteins
9. Albumin
10. Albumin and Globulin Ratio

## Preprocessing
A robust `scikit-learn` Pipeline is implemented to avoid data leakage. The preprocessing steps include:
- Imputing missing values using the median strategy.
- Scaling numerical features using `StandardScaler`.
- Gender and Target label encoding.

## Feature Selection
Advanced statistical feature selection is performed using `SelectKBest` with `mutual_info_classif` to select the top features before model training.

## Machine Learning Models
### Gradient Boosting
A `GradientBoostingClassifier` trained to handle sequential error optimization.

### AdaBoost
An `AdaBoostClassifier` built to adaptively boost weak learners focusing on misclassified instances.

### Ensemble Model
A `VotingClassifier` (soft voting) combines the probabilistic predictions of both Gradient Boosting and AdaBoost to form a final robust model.

## K-fold Cross Validation
The model evaluation process uses `StratifiedKFold` (5 splits) to ensure balanced class distributions across all folds during cross-validation. 

## Evaluation Metrics
The final model is evaluated on a dedicated test set (20% split) reporting:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Installation
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Training
To train the model, perform feature selection, evaluate through cross-validation, and save the pipeline, run:
```bash
python train_model.py
```
This script will produce `models/liver_disease_model.joblib`.

## Running Streamlit
To launch the user-friendly interface:
```bash
streamlit run app.py
```
The application allows clinical values to be inputted to predict the presence of liver disease.
