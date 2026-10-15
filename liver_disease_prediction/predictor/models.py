import uuid
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


def generate_assessment_id():
    """Generates a human-friendly unique assessment reference ID, e.g. LC-A1B2C3D4"""
    return f"LC-{uuid.uuid4().hex[:8].upper()}"


class PredictionRecord(models.Model):
    """
    Stores individual risk assessment inputs, predictions, and model metadata.
    Educational demonstration record - contains no PII.
    """
    CLASS_NO_DISEASE = 0
    CLASS_DISEASE = 1

    PREDICTION_CHOICES = [
        (CLASS_NO_DISEASE, 'Negative (No Liver Disease)'),
        (CLASS_DISEASE, 'Positive (Liver Disease)'),
    ]

    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    assessment_id = models.CharField(
        max_length=32,
        unique=True,
        default=generate_assessment_id,
        editable=False,
        db_index=True,
        help_text="Unique assessment reference ID"
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='predictions',
        help_text="User who initiated the assessment (if authenticated)"
    )

    # 10 clinical features
    age = models.PositiveIntegerField(help_text="Age in years (1-120)")
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, help_text="Biological sex")
    total_bilirubin = models.FloatField(help_text="Total Bilirubin (mg/dL)")
    direct_bilirubin = models.FloatField(help_text="Direct Bilirubin (mg/dL)")
    alkaline_phosphotase = models.FloatField(help_text="Alkaline Phosphotase (IU/L)")
    alamine_aminotransferase = models.FloatField(help_text="Alamine Aminotransferase (SGPT) (IU/L)")
    aspartate_aminotransferase = models.FloatField(help_text="Aspartate Aminotransferase (SGOT) (IU/L)")
    total_proteins = models.FloatField(help_text="Total Proteins (g/dL)")
    albumin = models.FloatField(help_text="Albumin (g/dL)")
    albumin_and_globulin_ratio = models.FloatField(help_text="Albumin and Globulin Ratio (A/G Ratio)")

    # Model Output
    predicted_class = models.IntegerField(
        choices=PREDICTION_CHOICES,
        help_text="0 = No Liver Disease, 1 = Liver Disease"
    )
    probability_disease = models.FloatField(
        null=True,
        blank=True,
        help_text="Model-estimated probability/score for Liver Disease (Class 1)"
    )
    probability_no_disease = models.FloatField(
        null=True,
        blank=True,
        help_text="Model-estimated probability/score for No Disease (Class 0)"
    )
    model_version = models.CharField(
        max_length=100,
        default="VotingClassifier (GBM+AdaBoost) k=8",
        help_text="Trained model artifact identifier"
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Record'
        verbose_name_plural = 'Prediction Records'

    def __str__(self):
        label = "Positive" if self.predicted_class == self.CLASS_DISEASE else "Negative"
        return f"{self.assessment_id} - {label} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    @property
    def is_positive(self):
        return self.predicted_class == self.CLASS_DISEASE

    @property
    def result_label(self):
        if self.predicted_class == self.CLASS_DISEASE:
            return "Positive for Liver Disease Class"
        return "Negative for Liver Disease Class"

    @property
    def result_statement(self):
        if self.predicted_class == self.CLASS_DISEASE:
            return "Model flagged this assessment as positive for the liver disease class."
        return "Model returned a negative classification for the liver disease class."

    @property
    def confidence_display(self):
        if self.probability_disease is not None and self.probability_no_disease is not None:
            active_score = self.probability_disease if self.is_positive else self.probability_no_disease
            return f"{active_score * 100:.1f}%"
        return "N/A"

    def get_features_dict(self):
        """Returns ordered dict of features with human labels and units"""
        return {
            'Age': {'label': 'Age', 'value': self.age, 'unit': 'years'},
            'Gender': {'label': 'Gender', 'value': self.gender, 'unit': ''},
            'Total_Bilirubin': {'label': 'Total Bilirubin', 'value': self.total_bilirubin, 'unit': 'mg/dL'},
            'Direct_Bilirubin': {'label': 'Direct Bilirubin', 'value': self.direct_bilirubin, 'unit': 'mg/dL'},
            'Alkaline_Phosphotase': {'label': 'Alkaline Phosphotase', 'value': self.alkaline_phosphotase, 'unit': 'IU/L'},
            'Alamine_Aminotransferase': {'label': 'Alamine Aminotransferase (SGPT)', 'value': self.alamine_aminotransferase, 'unit': 'IU/L'},
            'Aspartate_Aminotransferase': {'label': 'Aspartate Aminotransferase (SGOT)', 'value': self.aspartate_aminotransferase, 'unit': 'IU/L'},
            'Total_Proteins': {'label': 'Total Proteins', 'value': self.total_proteins, 'unit': 'g/dL'},
            'Albumin': {'label': 'Albumin', 'value': self.albumin, 'unit': 'g/dL'},
            'Albumin_and_Globulin_Ratio': {'label': 'Albumin and Globulin Ratio', 'value': self.albumin_and_globulin_ratio, 'unit': 'ratio'},
        }

    def get_absolute_url(self):
        return reverse('predictor:prediction_detail', kwargs={'assessment_id': self.assessment_id})
