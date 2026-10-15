# LiverCare AI — Liver Disease Risk Assessment System

A professional, enterprise-grade clinical decision support web application built with **Django 5** and **scikit-learn**, converting the legacy Streamlit prototype into a full-featured medical analytics platform.

The system utilizes the pre-trained soft-voting ensemble model trained on the **UCI Indian Liver Patient Dataset (ILPD)** to assess liver disease risk across ten key clinical laboratory biomarkers.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Machine Learning Pipeline & Preserved Artifact](#machine-learning-pipeline--preserved-artifact)
3. [Verified Model Evaluation Metrics](#verified-model-evaluation-metrics)
4. [Technology Stack](#technology-stack)
5. [Project Architecture](#project-architecture)
6. [Quickstart & Setup Instructions (Windows PowerShell)](#quickstart--setup-instructions-windows-powershell)
7. [Running the Test Suite](#running-the-test-suite)
8. [Reproducing Verified Evaluation Metrics](#reproducing-verified-evaluation-metrics)
9. [Clinical Staff Authentication](#clinical-staff-authentication)
10. [Backup Streamlit Interface](#backup-streamlit-interface)
11. [Educational & Medical Disclaimer](#educational--medical-disclaimer)

---

## Key Features

- **Modern Medical Dashboard**: Real-time summary statistics tracking actual application assessments, model prediction distribution (Chart.js doughnut visualization), quick-action links, and recent audit logs.
- **Dedicated 10-Feature Assessment Form**: Organizes the 10 physiological biomarkers into logical clinical sections (Patient Demographics, Bilirubin Levels, Liver Enzymes, and Protein Concentrations) with client-side and server-side range validation.
- **In-Memory Singleton Prediction Service**: Loads `models/liver_disease_model.joblib` once per worker process, formats exact feature vectors in the correct schema and order, and safely computes soft-voting probabilities without modifying pipeline artifacts.
- **Printable Clinical Assessment Reports**: Comprehensive individual report views complete with reference IDs, assessment timestamps, biomarker breakdowns, model confidence estimates, and print stylesheets (`@media print`).
- **Secure Historical Audit Log**: Search by reference ID, filter by predicted class, sort by date or patient age, with pagination and one-click JSON data export.
- **Role-Based Privacy Protection**: Restricts sensitive assessment history to authenticated clinical staff while permitting anonymous patients to view their immediate assessment outcome in-session.
- **One-Click Clinician Demo Login**: Facilitates instant testing and auditing for reviewers without requiring manual user creation.
- **Reproducible Evaluation**: Custom Django management command (`python manage.py evaluate_model`) to re-verify accuracy, precision, recall, F1, and 5-fold cross-validation on the local dataset.

---

## Machine Learning Pipeline & Preserved Artifact

The application reuses the trained scikit-learn pipeline saved at `models/liver_disease_model.joblib`:

1. **Missing-Value Imputation**: `SimpleImputer(strategy='median')` handling missing biomarkers (e.g. `Albumin_and_Globulin_Ratio`) without data leakage.
2. **Feature Scaling**: `StandardScaler()` standardizing numerical distributions.
3. **Statistical Feature Selection**: `SelectKBest(score_func=mutual_info_classif, k=8)` selecting the 8 most discriminative biomarkers:
   - `Age`
   - `Gender`
   - `Total_Bilirubin`
   - `Direct_Bilirubin`
   - `Alkaline_Phosphotase`
   - `Alamine_Aminotransferase`
   - `Aspartate_Aminotransferase`
   - `Albumin_and_Globulin_Ratio`
4. **Ensemble Architecture**: Soft-voting `VotingClassifier` combining:
   - `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)`
   - `AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)`
5. **Class Label Mapping**:
   - `1` = Positive (Liver Disease)
   - `0` = Negative (No Liver Disease)
6. **Gender Encoding**:
   - `Male` = 1
   - `Female` = 0

---

## Verified Model Evaluation Metrics

Evaluated on an 80/20 stratified split (117 test samples) of the UCI Indian Liver Patient Dataset:

| Metric | Verified Score | Description |
| :--- | :--- | :--- |
| **Test Accuracy** | **88.89%** (0.8889) | Overall proportion of correct predictions |
| **Precision (Class 1)** | **88.04%** (0.8804) | True positives out of positive model flags |
| **Recall / Sensitivity** | **97.59%** (0.9759) | Proportion of true disease cases detected |
| **F1-Score** | **92.57%** (0.9257) | Harmonic mean of precision and recall |
| **5-Fold Cross Validation** | **86.69% (&plusmn; 2.22%)** | Mean stratified 5-fold generalization accuracy |

### Confusion Matrix (Test Split n=117)

```
                 Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        23 (TN)                   11 (FP)
Actual Positive (1)         2 (FN)                   81 (TP)
```

---

## Technology Stack

- **Backend**: Python 3.11+, Django 5.2+, Pandas 3.0+, NumPy 2.3+, Scikit-learn 1.8+, Joblib 1.5+.
- **Database**: SQLite (default lightweight local storage for assessment records and metadata).
- **Frontend**: Django Server-Rendered Templates, HTML5, CSS3, Vanilla JavaScript, Bootstrap 5.3, Bootstrap Icons 1.11, Chart.js 4.4.
- **Environment Management**: `python-dotenv`.

---

## Project Architecture

```text
liver_disease_prediction/
├── manage.py                          # Django management runner
├── requirements.txt                   # Project dependencies
├── .env.example                       # Environment configuration template
├── README.md                          # Documentation and instructions
├── livercare/                         # Django project core
│   ├── __init__.py
│   ├── settings.py                    # Environment, security, logging, static & ML paths
│   ├── urls.py                        # Root URL router
│   ├── wsgi.py                        # WSGI entrypoint
│   └── asgi.py                        # ASGI entrypoint
├── predictor/                         # Application package
│   ├── __init__.py
│   ├── models.py                      # PredictionRecord model & helper properties
│   ├── forms.py                       # LiverPredictionForm with medical validation
│   ├── views.py                       # Dashboard, prediction, results, history, auth views
│   ├── urls.py                        # URL patterns
│   ├── admin.py                       # Django admin configuration
│   ├── tests.py                       # Comprehensive 19 automated unit & integration tests
│   ├── management/
│   │   └── commands/
│   │       └── evaluate_model.py      # Management command for metric verification
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py      # Thread-safe pipeline loader & inference engine
│   ├── templates/
│   │   └── predictor/
│   │       ├── base.html              # Medical shell, sidebar navigation, disclaimer
│   │       ├── dashboard.html         # Analytics dashboard & Chart.js distribution
│   │       ├── prediction_form.html   # Clinical 10-feature assessment form
│   │       ├── prediction_result.html # Detailed assessment result & print report
│   │       ├── prediction_history.html# Filterable, paginated audit records
│   │       ├── prediction_detail.html # Detail examination & JSON export
│   │       ├── model_info.html        # ML architecture & verified benchmark display
│   │       └── login.html             # Clinician login & 1-click demo button
│   └── static/
│       └── predictor/
│           ├── css/
│           │   └── style.css          # Medical navy/teal design tokens & print styles
│           └── js/
│               └── main.js            # Sidebar toggle, loading overlay, alert management
├── models/
│   └── liver_disease_model.joblib     # Preserved trained scikit-learn pipeline
├── data/
│   ├── dataset.csv                    # UCI Indian Liver Patient Dataset
│   └── evaluation_metrics.json        # Verified benchmarks generated by CLI command
├── app.py                             # Preserved Streamlit backup application
├── model_utils.py                     # Legacy pipeline building utilities
├── train_model.py                     # Training script
├── tune_data.py                       # Data preparation script
└── test_data.py                       # Data schema test script
```

---

## Quickstart & Setup Instructions (Windows PowerShell)

Open Windows PowerShell in the project directory:

```powershell
cd c:\Users\Harsha\Downloads\projectt\liver_disease_prediction
```

### 1. Create and Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

*(If script execution is disabled on your machine, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Initialize Environment Variables

```powershell
Copy-Item .env.example .env
```

### 4. Run Migrations & Metric Verification

```powershell
python manage.py migrate
python manage.py evaluate_model --save
```

### 5. Start the Development Server

```powershell
python manage.py runserver
```

Open your browser at:

**`http://127.0.0.1:8000/`**

---

## Running the Test Suite

The test suite contains 19 automated tests verifying URL resolution, form validation, bounds checking, negative value rejection, column ordering, gender encoding, model prediction, database persistence, access control, and JSON exports.

To run tests:

```powershell
python manage.py test
```

Expected result:
```text
Ran 19 tests in ~25s
OK
```

---

## Reproducing Verified Evaluation Metrics

To reproduce the exact classification metrics from the dataset without retraining or modifying the saved model:

```powershell
python manage.py evaluate_model --save
```

This calculates test accuracy, precision, recall, F1, 5-fold cross-validation scores, and the confusion matrix, writing the verified output to `data/evaluation_metrics.json`.

---

## Clinical Staff Authentication

Access to full patient assessment history is restricted to authenticated users.

1. Navigate to `http://127.0.0.1:8000/login/`.
2. Click **"1-Click Clinician Demo Login"** for instant testing, OR
3. Create a superuser manually:
   ```powershell
   python manage.py createsuperuser
   ```
4. Access the Django admin interface at:
   `http://127.0.0.1:8000/admin/`

---

## Backup Streamlit Interface

The original Streamlit application remains intact as an alternate prototype:

```powershell
streamlit run app.py
```

---

## Educational & Medical Disclaimer

> **IMPORTANT**: This application is an educational and research machine learning demonstration. Its predictions are based on statistical pattern matching from retrospective data and **do not constitute a medical diagnosis, clinical verdict, or treatment advice**. Never use this system as a substitute for professional clinical judgment, laboratory confirmation, or physician evaluation.
