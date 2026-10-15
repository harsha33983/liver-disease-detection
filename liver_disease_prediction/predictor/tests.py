import json
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
import pandas as pd
import numpy as np

from predictor.models import PredictionRecord
from predictor.forms import LiverPredictionForm
from predictor.services import (
    PredictionService,
    ModelLoadError,
    FEATURE_NAMES,
)


class TestUrlsAndRouting(TestCase):
    """Verifies that all named URLs resolve and reverse correctly."""

    def test_url_reversal(self):
        self.assertEqual(reverse('predictor:dashboard'), '/')
        self.assertEqual(reverse('predictor:predict'), '/predict/')
        self.assertEqual(reverse('predictor:prediction_history'), '/history/')
        self.assertEqual(reverse('predictor:model_info'), '/model-info/')
        self.assertEqual(reverse('predictor:login'), '/login/')
        self.assertEqual(reverse('predictor:logout'), '/logout/')
        self.assertEqual(
            reverse('predictor:prediction_result', kwargs={'assessment_id': 'LC-TEST1234'}),
            '/result/LC-TEST1234/'
        )
        self.assertEqual(
            reverse('predictor:prediction_detail', kwargs={'assessment_id': 'LC-TEST1234'}),
            '/history/LC-TEST1234/'
        )


class TestFormValidation(TestCase):
    """Validates patient demographic and biomarker input criteria."""

    def setUp(self):
        self.valid_data = {
            'Age': 45,
            'Gender': 'Male',
            'Total_Bilirubin': 1.2,
            'Direct_Bilirubin': 0.4,
            'Alkaline_Phosphotase': 200.0,
            'Alamine_Aminotransferase': 35.0,
            'Aspartate_Aminotransferase': 40.0,
            'Total_Proteins': 7.0,
            'Albumin': 3.5,
            'Albumin_and_Globulin_Ratio': 1.0,
        }

    def test_valid_form(self):
        form = LiverPredictionForm(data=self.valid_data)
        self.assertTrue(form.is_valid(), form.errors)

    def test_missing_required_fields(self):
        incomplete_data = self.valid_data.copy()
        del incomplete_data['Age']
        del incomplete_data['Total_Bilirubin']
        form = LiverPredictionForm(data=incomplete_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Age', form.errors)
        self.assertIn('Total_Bilirubin', form.errors)

    def test_negative_laboratory_values_rejected(self):
        invalid_data = self.valid_data.copy()
        invalid_data['Total_Bilirubin'] = -2.5
        form = LiverPredictionForm(data=invalid_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Total_Bilirubin', form.errors)

    def test_invalid_age_bounds(self):
        # Age exceeds 120
        invalid_data = self.valid_data.copy()
        invalid_data['Age'] = 150
        form = LiverPredictionForm(data=invalid_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Age', form.errors)

        # Age < 1
        invalid_data['Age'] = 0
        form2 = LiverPredictionForm(data=invalid_data)
        self.assertFalse(form2.is_valid())
        self.assertIn('Age', form2.errors)

    def test_direct_bilirubin_cannot_exceed_total_bilirubin(self):
        invalid_data = self.valid_data.copy()
        invalid_data['Total_Bilirubin'] = 0.5
        invalid_data['Direct_Bilirubin'] = 1.5  # Greater than TB
        form = LiverPredictionForm(data=invalid_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Direct_Bilirubin', form.errors)

    def test_albumin_cannot_exceed_total_proteins(self):
        invalid_data = self.valid_data.copy()
        invalid_data['Total_Proteins'] = 5.0
        invalid_data['Albumin'] = 6.5  # Greater than TP
        form = LiverPredictionForm(data=invalid_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Albumin', form.errors)


class TestPredictionService(TestCase):
    """Tests the machine learning prediction backend, feature order, and encoding."""

    def setUp(self):
        self.sample_input = {
            'Age': 50,
            'Gender': 'Female',
            'Total_Bilirubin': 2.0,
            'Direct_Bilirubin': 0.8,
            'Alkaline_Phosphotase': 240.0,
            'Alamine_Aminotransferase': 55.0,
            'Aspartate_Aminotransferase': 60.0,
            'Total_Proteins': 6.5,
            'Albumin': 3.0,
            'Albumin_and_Globulin_Ratio': 0.85,
        }

    def test_exact_feature_column_order(self):
        df = PredictionService.prepare_input_dataframe(self.sample_input)
        expected_columns = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
            'Albumin_and_Globulin_Ratio'
        ]
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(len(df.columns), 10)

    def test_gender_encoding(self):
        male_data = self.sample_input.copy()
        male_data['Gender'] = 'Male'
        df_male = PredictionService.prepare_input_dataframe(male_data)
        self.assertEqual(df_male['Gender'].iloc[0], 1)

        female_data = self.sample_input.copy()
        female_data['Gender'] = 'Female'
        df_female = PredictionService.prepare_input_dataframe(female_data)
        self.assertEqual(df_female['Gender'].iloc[0], 0)

    def test_actual_saved_model_prediction(self):
        # Ensure actual model loads and predicts without runtime errors
        result = PredictionService.predict(self.sample_input)
        self.assertTrue(result['success'])
        self.assertIn(result['predicted_class'], (0, 1))
        self.assertIn('probability_disease', result)
        self.assertIn('probability_no_disease', result)
        self.assertIn('result_label', result)
        self.assertIn('result_statement', result)

        if result['predicted_class'] == 1:
            self.assertEqual(result['result_statement'], "Model flagged this assessment as positive for the liver disease class.")
        else:
            self.assertEqual(result['result_statement'], "Model returned a negative classification for the liver disease class.")

    def test_missing_model_artifact_handling(self):
        with patch.object(PredictionService, 'get_model_path') as mock_path:
            mock_path.return_value = Path('/non/existent/model.joblib')
            with self.assertRaises(ModelLoadError):
                PredictionService.load_model(force_reload=True)
            # Restore model singleton
            PredictionService._model = None


class TestViewsAndWorkflow(TestCase):
    """End-to-end integration tests for views, database persistence, and sessions."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='doctor_test',
            password='TestPassword123!',
            is_staff=True
        )
        self.form_payload = {
            'Age': 48,
            'Gender': 'Male',
            'Total_Bilirubin': 1.8,
            'Direct_Bilirubin': 0.7,
            'Alkaline_Phosphotase': 260.0,
            'Alamine_Aminotransferase': 45.0,
            'Aspartate_Aminotransferase': 50.0,
            'Total_Proteins': 7.2,
            'Albumin': 3.6,
            'Albumin_and_Globulin_Ratio': 1.0,
        }

    def test_dashboard_view_renders(self):
        response = self.client.get(reverse('predictor:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictor/dashboard.html')
        self.assertIn('total_records', response.context)
        self.assertContains(response, "LiverCare AI")

    def test_prediction_form_submission_creates_record(self):
        initial_count = PredictionRecord.objects.count()
        response = self.client.post(reverse('predictor:predict'), data=self.form_payload, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionRecord.objects.count(), initial_count + 1)

        record = PredictionRecord.objects.latest('created_at')
        self.assertEqual(record.age, 48)
        self.assertEqual(record.gender, 'Male')
        self.assertIn(record.predicted_class, (0, 1))

        # Check redirected result page displays reference ID
        self.assertContains(response, record.assessment_id)
        self.assertContains(response, "Model-Estimated Score")

    def test_anonymous_access_to_own_assessment_allowed(self):
        # When an anonymous user submits, session holds last_assessment_id
        self.client.post(reverse('predictor:predict'), data=self.form_payload)
        record = PredictionRecord.objects.latest('created_at')
        
        res = self.client.get(reverse('predictor:prediction_result', kwargs={'assessment_id': record.assessment_id}))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, record.assessment_id)

    def test_anonymous_access_to_other_assessment_blocked(self):
        # Create record directly in DB without session
        record = PredictionRecord.objects.create(
            age=30,
            gender='Female',
            total_bilirubin=1.0,
            direct_bilirubin=0.3,
            alkaline_phosphotase=150.0,
            alamine_aminotransferase=20.0,
            aspartate_aminotransferase=25.0,
            total_proteins=6.5,
            albumin=3.2,
            albumin_and_globulin_ratio=0.9,
            predicted_class=0,
            probability_no_disease=0.8,
            probability_disease=0.2,
        )
        # Attempt to access without session or auth
        anon_client = Client()
        res = anon_client.get(reverse('predictor:prediction_result', kwargs={'assessment_id': record.assessment_id}))
        self.assertEqual(res.status_code, 302)
        self.assertIn(reverse('predictor:login'), res.url)

    def test_history_requires_authentication(self):
        anon_client = Client()
        res = anon_client.get(reverse('predictor:prediction_history'))
        self.assertEqual(res.status_code, 302)
        self.assertIn(reverse('predictor:login'), res.url)

    def test_authenticated_history_and_filters(self):
        # Create test records
        PredictionRecord.objects.create(
            assessment_id='LC-TESTPOS1',
            age=55,
            gender='Male',
            total_bilirubin=3.5,
            direct_bilirubin=1.5,
            alkaline_phosphotase=300.0,
            alamine_aminotransferase=80.0,
            aspartate_aminotransferase=90.0,
            total_proteins=6.0,
            albumin=2.8,
            albumin_and_globulin_ratio=0.7,
            predicted_class=1,
            probability_disease=0.85,
            probability_no_disease=0.15,
        )
        PredictionRecord.objects.create(
            assessment_id='LC-TESTNEG1',
            age=25,
            gender='Female',
            total_bilirubin=0.8,
            direct_bilirubin=0.2,
            alkaline_phosphotase=120.0,
            alamine_aminotransferase=18.0,
            aspartate_aminotransferase=22.0,
            total_proteins=7.5,
            albumin=4.0,
            albumin_and_globulin_ratio=1.1,
            predicted_class=0,
            probability_disease=0.1,
            probability_no_disease=0.9,
        )

        self.client.force_login(self.user)
        res = self.client.get(reverse('predictor:prediction_history'))
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'LC-TESTPOS1')
        self.assertContains(res, 'LC-TESTNEG1')

        # Filter by class=1
        res_pos = self.client.get(reverse('predictor:prediction_history') + '?class=1')
        self.assertContains(res_pos, 'LC-TESTPOS1')
        self.assertNotContains(res_pos, 'LC-TESTNEG1')

        # Search by assessment ID
        res_search = self.client.get(reverse('predictor:prediction_history') + '?search=TESTNEG1')
        self.assertContains(res_search, 'LC-TESTNEG1')
        self.assertNotContains(res_search, 'LC-TESTPOS1')

    def test_json_detail_export(self):
        record = PredictionRecord.objects.create(
            assessment_id='LC-JSONTEST',
            age=40,
            gender='Male',
            total_bilirubin=1.0,
            direct_bilirubin=0.2,
            alkaline_phosphotase=150.0,
            alamine_aminotransferase=25.0,
            aspartate_aminotransferase=30.0,
            total_proteins=6.8,
            albumin=3.2,
            albumin_and_globulin_ratio=0.9,
            predicted_class=0,
            probability_disease=0.2,
            probability_no_disease=0.8,
        )
        self.client.force_login(self.user)
        res = self.client.get(reverse('predictor:prediction_detail', kwargs={'assessment_id': record.assessment_id}) + '?format=json')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res['Content-Type'], 'application/json')
        data = json.loads(res.content)
        self.assertEqual(data['assessment_id'], 'LC-JSONTEST')
        self.assertEqual(data['predicted_class'], 0)
        self.assertIn('measurements', data)

    def test_model_info_view(self):
        res = self.client.get(reverse('predictor:model_info'))
        self.assertEqual(res.status_code, 200)
        self.assertTemplateUsed(res, 'predictor/model_info.html')
        self.assertContains(res, "UCI Indian Liver Patient Dataset")
        self.assertContains(res, "VotingClassifier")
