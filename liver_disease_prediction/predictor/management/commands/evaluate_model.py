import json
import logging
import os
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Safely evaluates the saved model pipeline against the dataset and generates verified metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--save',
            action='store_true',
            help='Save the computed metrics to data/evaluation_metrics.json for the web dashboard'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting model evaluation..."))

        model_path = Path(settings.BASE_DIR) / 'models' / 'liver_disease_model.joblib'
        data_path = Path(settings.BASE_DIR) / 'data' / 'dataset.csv'

        if not model_path.exists():
            self.stderr.write(self.style.ERROR(f"Model file not found at {model_path}"))
            return

        if not data_path.exists():
            self.stderr.write(self.style.ERROR(f"Dataset not found at {data_path}"))
            return

        import joblib
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        X = df.drop('Target', axis=1)
        y = df['Target'].copy()

        # Target mapping: 1 -> 1 (Disease), 2 -> 0 (No Disease)
        y = y.map({1: 1, 2: 0})
        # Gender mapping
        if 'Gender' in X.columns:
            X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})
            X['Gender'] = pd.to_numeric(X['Gender'])

        # Stratified split matching train_model.py
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True)

        # 5-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        metrics = {
            'dataset_name': 'UCI Indian Liver Patient Dataset (ILPD)',
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'cv_mean_accuracy': cv_mean,
            'cv_std_accuracy': cv_std,
            'cv_folds': [float(s) for s in cv_scores],
            'confusion_matrix': {
                'tn': cm[0][0],
                'fp': cm[0][1],
                'fn': cm[1][0],
                'tp': cm[1][1],
                'raw': cm
            },
            'classification_report': cr,
            'evaluated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        }

        self.stdout.write(self.style.SUCCESS(f"Evaluation Complete!"))
        self.stdout.write(f"Test Accuracy: {acc:.4f}")
        self.stdout.write(f"Precision:     {prec:.4f}")
        self.stdout.write(f"Recall:        {rec:.4f}")
        self.stdout.write(f"F1 Score:      {f1:.4f}")
        self.stdout.write(f"5-Fold CV:     {cv_mean:.4f} (+/- {cv_std:.4f})")
        self.stdout.write(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

        output_path = Path(settings.BASE_DIR) / 'data' / 'evaluation_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Verified metrics written to {output_path}"))
