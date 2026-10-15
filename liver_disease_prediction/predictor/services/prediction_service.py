import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

# Expected 10 feature names in exact order required by the trained pipeline
FEATURE_NAMES = [
    'Age',
    'Gender',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Proteins',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
]


class ModelLoadError(Exception):
    """Raised when the trained machine learning pipeline cannot be loaded."""
    pass


class PredictionExecutionError(Exception):
    """Raised when an error occurs during model inference."""
    pass


class PredictionService:
    """
    Thread-safe, cached service for loading the saved scikit-learn pipeline
    and running liver disease risk inferences.
    """
    _model = None
    _model_path: Optional[Path] = None

    @classmethod
    def get_model_path(cls) -> Path:
        """Resolves the absolute path to the saved joblib model artifact."""
        if hasattr(settings, 'LIVER_MODEL_PATH'):
            return Path(settings.LIVER_MODEL_PATH)
        return Path(settings.BASE_DIR) / 'models' / 'liver_disease_model.joblib'

    @classmethod
    def load_model(cls, force_reload: bool = False):
        """
        Loads the pre-trained Joblib pipeline once per process.
        """
        if cls._model is not None and not force_reload:
            return cls._model

        model_path = cls.get_model_path()
        cls._model_path = model_path

        if not model_path.exists():
            error_msg = f"Model artifact not found at configured path."
            logger.error(f"PredictionService: Model file missing at {model_path}")
            raise ModelLoadError("Trained model artifact is missing from the server.")

        try:
            logger.info(f"Loading trained pipeline from {model_path}...")
            model = joblib.load(model_path)
            cls._model = model
            logger.info("Successfully loaded liver disease prediction pipeline.")
            return cls._model
        except Exception as exc:
            logger.exception("Failed to deserialize model artifact.")
            raise ModelLoadError("Failed to initialize the machine learning pipeline.") from exc

    @classmethod
    def is_model_available(cls) -> bool:
        """Checks whether the model artifact is available on disk."""
        try:
            path = cls.get_model_path()
            return path.is_file()
        except Exception:
            return False

    @classmethod
    def prepare_input_dataframe(cls, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates, casts, and encodes the 10 features into a Pandas DataFrame
        matching the exact feature names and order required by the model.
        """
        gender_val = raw_data.get('Gender')
        if isinstance(gender_val, str):
            gender_normalized = gender_val.strip().lower()
            if gender_normalized in ('1', 'male', 'm'):
                gender_encoded = 1
            elif gender_normalized in ('0', 'female', 'f'):
                gender_encoded = 0
            else:
                raise ValueError(f"Invalid gender specification: {gender_val}")
        elif isinstance(gender_val, (int, float)):
            gender_encoded = int(gender_val)
        else:
            raise ValueError("Gender must be specified as 'Male' or 'Female'.")

        # Extract values ensuring strict casting and presence
        try:
            features = {
                'Age': [float(raw_data['Age'])],
                'Gender': [gender_encoded],
                'Total_Bilirubin': [float(raw_data['Total_Bilirubin'])],
                'Direct_Bilirubin': [float(raw_data['Direct_Bilirubin'])],
                'Alkaline_Phosphotase': [float(raw_data['Alkaline_Phosphotase'])],
                'Alamine_Aminotransferase': [float(raw_data['Alamine_Aminotransferase'])],
                'Aspartate_Aminotransferase': [float(raw_data['Aspartate_Aminotransferase'])],
                'Total_Proteins': [float(raw_data['Total_Proteins'])],
                'Albumin': [float(raw_data['Albumin'])],
                'Albumin_and_Globulin_Ratio': [float(raw_data['Albumin_and_Globulin_Ratio'])],
            }
        except KeyError as ke:
            raise ValueError(f"Missing required clinical feature: {ke.args[0]}")
        except (ValueError, TypeError) as ve:
            raise ValueError("All clinical laboratory values must be valid real numbers.")

        # Construct DataFrame with exact column order
        df = pd.DataFrame(features, columns=FEATURE_NAMES)

        # Sanity check for NaNs or Infinities in constructed DataFrame
        if df.isna().any().any():
            raise ValueError("Input data contains missing or NaN values.")
        if np.isinf(df.to_numpy(dtype=float)).any():
            raise ValueError("Input data contains infinite values.")

        return df

    @classmethod
    def predict(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes model prediction and extracts class probabilities.
        Returns a structured dictionary with sanitized result metadata.
        """
        try:
            model = cls.load_model()
            input_df = cls.prepare_input_dataframe(raw_data)

            # 1. Run pipeline predict
            raw_pred = model.predict(input_df)
            predicted_class = int(raw_pred[0])

            # 2. Check for predict_proba
            prob_disease = None
            prob_no_disease = None
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(input_df)[0]
                    # Classes in pipeline are [0, 1]
                    # Index 0 corresponds to Class 0 (No Disease)
                    # Index 1 corresponds to Class 1 (Disease)
                    classes = list(getattr(model, 'classes_', [0, 1]))
                    if 0 in classes and 1 in classes:
                        idx_0 = classes.index(0)
                        idx_1 = classes.index(1)
                        prob_no_disease = float(probas[idx_0])
                        prob_disease = float(probas[idx_1])
                    else:
                        prob_no_disease = float(probas[0])
                        prob_disease = float(probas[1]) if len(probas) > 1 else (1.0 - prob_no_disease)
                except Exception as pe:
                    logger.warning(f"Could not extract class probabilities: {pe}")

            # 3. Formulate structured response
            is_disease = (predicted_class == 1)
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'is_positive': is_disease,
                'result_label': "Positive for Liver Disease Class" if is_disease else "Negative for Liver Disease Class",
                'result_statement': (
                    "Model flagged this assessment as positive for the liver disease class."
                    if is_disease else
                    "Model returned a negative classification for the liver disease class."
                ),
                'probability_disease': prob_disease,
                'probability_no_disease': prob_no_disease,
                'confidence_display': f"{prob_disease * 100:.1f}%" if is_disease and prob_disease else (
                    f"{prob_no_disease * 100:.1f}%" if not is_disease and prob_no_disease else "N/A"
                ),
                'model_version': "VotingClassifier (GradientBoosting + AdaBoost) k=8",
                'feature_names': FEATURE_NAMES,
            }
            return result

        except (ModelLoadError, ValueError) as client_e:
            logger.warning(f"Prediction validation/model error: {client_e}")
            raise
        except Exception as exc:
            logger.exception("Unexpected error during model inference execution.")
            raise PredictionExecutionError(
                "An unexpected internal error occurred while processing the clinical assessment."
            ) from exc
