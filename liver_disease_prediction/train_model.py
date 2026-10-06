import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import model_utils

def main():
    print("Loading dataset...")
    X, y = model_utils.load_dataset()
    if X is None:
        return

    print("Cleaning and encoding data...")
    X, y = model_utils.clean_and_encode(X, y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = model_utils.create_preprocessing_pipeline()

    np.random.seed(42)
    feature_selection = model_utils.get_feature_selection(k=8)

    gb_model = model_utils.build_gradient_boosting()
    ada_model = model_utils.build_adaboost()
    ensemble = model_utils.build_ensemble(gb_model, ada_model)

    pipeline = model_utils.create_full_pipeline(preprocessor, feature_selection, ensemble)

    print("Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    np.random.seed(42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Fold Accuracies: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"CV Standard Deviation: {cv_scores.std():.4f}")

    print("\nTraining the final pipeline on the training set...")
    np.random.seed(42)
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    selected_mask = pipeline.named_steps['feature_selection'].get_support()
    selected_features = X.columns[selected_mask]
    print("\nSelected Features:")
    for feature in selected_features:
        print(f"- {feature}")

    model_path = 'models/liver_disease_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nModel successfully saved to {model_path}")

if __name__ == "__main__":
    main()
