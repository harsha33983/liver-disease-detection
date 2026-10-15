import json
import logging
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from .forms import LiverPredictionForm
from .models import PredictionRecord
from .services import (
    PredictionService,
    ModelLoadError,
    PredictionExecutionError,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


def dashboard_view(request):
    """
    Primary medical analytics dashboard.
    Displays actual application usage counts, prediction distribution,
    model status, clinical feature catalog, and recent assessments.
    """
    total_records = PredictionRecord.objects.count()
    positive_count = PredictionRecord.objects.filter(predicted_class=PredictionRecord.CLASS_DISEASE).count()
    negative_count = PredictionRecord.objects.filter(predicted_class=PredictionRecord.CLASS_NO_DISEASE).count()

    positive_percent = round((positive_count / total_records * 100), 1) if total_records > 0 else 0
    negative_percent = round((negative_count / total_records * 100), 1) if total_records > 0 else 0

    recent_predictions = PredictionRecord.objects.all()[:6]
    model_available = PredictionService.is_model_available()

    # Load verified evaluation summary if present
    metrics_file = Path(settings.BASE_DIR) / 'data' / 'evaluation_metrics.json'
    verified_metrics = None
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                verified_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read evaluation metrics file: {e}")

    context = {
        'total_records': total_records,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'recent_predictions': recent_predictions,
        'model_available': model_available,
        'model_version': "VotingClassifier (GBM + AdaBoost) k=8",
        'verified_metrics': verified_metrics,
    }
    return render(request, 'predictor/dashboard.html', context)


def predict_view(request):
    """
    Dedicated clinical risk assessment form view.
    Validates the 10 features, calls the pre-trained pipeline, and persists the record.
    """
    model_available = PredictionService.is_model_available()

    if request.method == 'POST':
        form = LiverPredictionForm(request.POST)
        if form.is_valid():
            try:
                cleaned_data = form.cleaned_data

                # Call Prediction Service with exact feature mapping
                prediction_result = PredictionService.predict(cleaned_data)

                # Persist to SQLite database
                record = PredictionRecord.objects.create(
                    created_by=request.user if request.user.is_authenticated else None,
                    age=cleaned_data['Age'],
                    gender=cleaned_data['Gender'],
                    total_bilirubin=cleaned_data['Total_Bilirubin'],
                    direct_bilirubin=cleaned_data['Direct_Bilirubin'],
                    alkaline_phosphotase=cleaned_data['Alkaline_Phosphotase'],
                    alamine_aminotransferase=cleaned_data['Alamine_Aminotransferase'],
                    aspartate_aminotransferase=cleaned_data['Aspartate_Aminotransferase'],
                    total_proteins=cleaned_data['Total_Proteins'],
                    albumin=cleaned_data['Albumin'],
                    albumin_and_globulin_ratio=cleaned_data['Albumin_and_Globulin_Ratio'],
                    predicted_class=prediction_result['predicted_class'],
                    probability_disease=prediction_result.get('probability_disease'),
                    probability_no_disease=prediction_result.get('probability_no_disease'),
                    model_version=prediction_result.get('model_version', 'VotingClassifier v1.0'),
                )

                # Save assessment ID in session for immediate anonymous review
                request.session['last_assessment_id'] = record.assessment_id

                messages.success(request, f"Assessment {record.assessment_id} processed successfully.")
                return redirect('predictor:prediction_result', assessment_id=record.assessment_id)

            except ModelLoadError as mle:
                logger.error(f"Prediction failed due to ModelLoadError: {mle}")
                messages.error(
                    request,
                    "The machine learning model artifact is currently unavailable. Please verify the model installation."
                )
            except ValueError as ve:
                logger.warning(f"Validation error in prediction pipeline: {ve}")
                messages.error(request, f"Input validation error: {str(ve)}")
            except PredictionExecutionError as pee:
                logger.error(f"Inference execution error: {pee}")
                messages.error(
                    request,
                    "An error occurred while evaluating the clinical measurements. Please check your inputs and try again."
                )
            except Exception as e:
                logger.exception("Unexpected error handling assessment submission.")
                messages.error(
                    request,
                    "A server error occurred while processing the assessment. Please contact system support."
                )
        else:
            messages.error(request, "Please correct the highlighted validation errors below.")
    else:
        form = LiverPredictionForm()

    context = {
        'form': form,
        'model_available': model_available,
    }
    return render(request, 'predictor/prediction_form.html', context)


def prediction_result_view(request, assessment_id):
    """
    Displays the results of a specific clinical assessment.
    Accessible to the creator or authenticated staff.
    """
    record = get_object_or_404(PredictionRecord, assessment_id=assessment_id)

    # Permission check: Allowed if authenticated user OR if this is the session's active assessment
    last_assessment = request.session.get('last_assessment_id')
    if not request.user.is_authenticated and last_assessment != assessment_id:
        messages.info(request, "Please log in with clinical credentials to view historical assessment records.")
        return redirect(f"{reverse('predictor:login')}?next={request.path}")

    features_dict = record.get_features_dict()

    context = {
        'record': record,
        'features_dict': features_dict,
        'is_print': request.GET.get('print') == 'true',
    }
    return render(request, 'predictor/prediction_result.html', context)


@login_required
def prediction_history_view(request):
    """
    Secure prediction history view.
    Restricted to authenticated staff/clinicians.
    Supports search, class filtering, sorting, and pagination.
    """
    records = PredictionRecord.objects.all()

    # Search by assessment ID
    query = request.GET.get('search', '').strip()
    if query:
        records = records.filter(Q(assessment_id__icontains=query))

    # Filter by predicted class
    class_filter = request.GET.get('class')
    if class_filter in ('0', '1'):
        records = records.filter(predicted_class=int(class_filter))

    # Sort
    sort_by = request.GET.get('sort', '-created_at')
    if sort_by in ('-created_at', 'created_at', 'age', '-age'):
        records = records.order_by(sort_by)
    else:
        records = records.order_by('-created_at')

    # Pagination
    paginator = Paginator(records, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Summary counts
    total_count = records.count()
    pos_count = records.filter(predicted_class=PredictionRecord.CLASS_DISEASE).count()
    neg_count = records.filter(predicted_class=PredictionRecord.CLASS_NO_DISEASE).count()

    context = {
        'page_obj': page_obj,
        'query': query,
        'class_filter': class_filter,
        'sort_by': sort_by,
        'total_count': total_count,
        'pos_count': pos_count,
        'neg_count': neg_count,
    }
    return render(request, 'predictor/prediction_history.html', context)


@login_required
def prediction_detail_view(request, assessment_id):
    """
    Detailed assessment examination and export view.
    """
    record = get_object_or_404(PredictionRecord, assessment_id=assessment_id)
    features_dict = record.get_features_dict()

    # Support JSON export
    if request.GET.get('format') == 'json':
        data = {
            'assessment_id': record.assessment_id,
            'created_at': record.created_at.isoformat(),
            'predicted_class': record.predicted_class,
            'result_label': record.result_label,
            'probability_disease': record.probability_disease,
            'probability_no_disease': record.probability_no_disease,
            'model_version': record.model_version,
            'measurements': {k: v['value'] for k, v in features_dict.items()},
            'disclaimer': "Educational machine learning assessment. Not a medical diagnosis."
        }
        return JsonResponse(data, json_dumps_params={'indent': 2})

    context = {
        'record': record,
        'features_dict': features_dict,
        'is_print': request.GET.get('print') == 'true',
    }
    return render(request, 'predictor/prediction_detail.html', context)


def model_info_view(request):
    """
    Displays the technical ML pipeline architecture, data preprocessing stages,
    feature selection insights, and verified evaluation metrics.
    """
    metrics_file = Path(settings.BASE_DIR) / 'data' / 'evaluation_metrics.json'
    verified_metrics = None
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                verified_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read evaluation metrics file: {e}")

    # Pipeline description dictionary
    pipeline_info = {
        'dataset_name': "UCI Indian Liver Patient Dataset (ILPD)",
        'dataset_source': "UCI Machine Learning Repository (ID 225)",
        'imputation': "Median Imputation (SimpleImputer) to handle missing values without data leakage",
        'scaling': "StandardScaler (zero mean, unit variance)",
        'feature_selection': "SelectKBest with Mutual Information Classification (mutual_info_classif, k=8)",
        'selected_features': [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Albumin_and_Globulin_Ratio'
        ],
        'unselected_features': ['Total_Proteins', 'Albumin'],
        'model_architecture': "Soft-Voting Ensemble (VotingClassifier)",
        'sub_models': [
            {
                'name': "Gradient Boosting Classifier",
                'hyperparameters': "n_estimators=100, learning_rate=0.1, random_state=42",
                'description': "Sequentially optimizes decision trees focusing on pseudo-residuals."
            },
            {
                'name': "AdaBoost Classifier",
                'hyperparameters': "n_estimators=100, learning_rate=1.0, random_state=42",
                'description': "Adaptively boosts weak decision stumps by increasing weights on misclassified instances."
            }
        ],
        'cross_validation': "5-Fold Stratified Cross-Validation (StratifiedKFold, shuffle=True, random_state=42)",
        'split_ratio': "80% Training (Stratified), 20% Held-out Test Set",
    }

    context = {
        'pipeline_info': pipeline_info,
        'verified_metrics': verified_metrics,
        'features_list': FEATURE_NAMES,
    }
    return render(request, 'predictor/model_info.html', context)


def login_view(request):
    """
    Authentication view for clinical staff with helper demo-login functionality.
    """
    if request.user.is_authenticated:
        return redirect('predictor:prediction_history')

    if request.method == 'POST':
        # Handle demo quick-login button
        if 'demo_login' in request.POST:
            demo_user, _ = User.objects.get_or_create(
                username='clinician_demo',
                defaults={
                    'email': 'clinician@livercare.local',
                    'first_name': 'Clinician',
                    'last_name': 'Demo',
                    'is_staff': True
                }
            )
            login(request, demo_user)
            messages.success(request, "Logged in as Clinician Demo.")
            next_url = request.GET.get('next') or request.POST.get('next') or reverse('predictor:prediction_history')
            return redirect(next_url)

        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}.")
            next_url = request.GET.get('next') or request.POST.get('next') or reverse('predictor:prediction_history')
            return redirect(next_url)
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    context = {
        'form': form,
        'next': request.GET.get('next', ''),
    }
    return render(request, 'predictor/login.html', context)


def logout_view(request):
    """Logs out the active user."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('predictor:dashboard')
