import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from ucimlrepo import fetch_ucirepo

def load_dataset():

    data_path = 'data/dataset.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X = df.drop('Target', axis=1)
        y = df['Target']
    else:
        ilpd = fetch_ucirepo(id=225)
        X = ilpd.data.features
        y = ilpd.data.targets

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

    col_mapping = {
        'Age': 'Age',
        'Gender': 'Gender',
        'TB': 'Total_Bilirubin',
        'DB': 'Direct_Bilirubin',
        'Alkphos': 'Alkaline_Phosphotase',
        'Sgpt': 'Alamine_Aminotransferase',
        'Sgot': 'Aspartate_Aminotransferase',
        'TP': 'Total_Proteins',
        'ALB': 'Albumin',
        'A/G Ratio': 'Albumin_and_Globulin_Ratio'
    }
    X = X.rename(columns=col_mapping)

    df = X.copy()
    df['Target'] = y
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    return X, y

def clean_and_encode(X, y):

    X_clean = X.copy()
    y_clean = y.copy()

    y_clean = y_clean.map({1: 1, 2: 0})

    if 'Gender' in X_clean.columns:
        X_clean['Gender'] = X_clean['Gender'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})
        X_clean['Gender'] = pd.to_numeric(X_clean['Gender'])

    return X_clean, y_clean

def create_preprocessing_pipeline():

    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

def get_feature_selection(k=8):

    return SelectKBest(score_func=mutual_info_classif, k=k)

def build_gradient_boosting():

    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

def build_adaboost():

    return AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)

def build_ensemble(gb_model, ada_model):

    return VotingClassifier(
        estimators=[
            ('gradient_boosting', gb_model),
            ('adaboost', ada_model)
        ],
        voting='soft'
    )

def create_full_pipeline(preprocessor, feature_selection, model):

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selection),
        ('model', model)
    ])
