import math
from django import forms
from .models import PredictionRecord


class LiverPredictionForm(forms.Form):
    """
    Clinical input form for the 10 liver disease biomarkers.
    Uses exact internal feature names required by the trained pipeline.
    """
    GENDER_CHOICES = [
        ('', '-- Select Biological Sex --'),
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    # Section 1: Patient Information
    Age = forms.IntegerField(
        label='Age',
        min_value=1,
        max_value=120,
        initial=45,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 45',
            'min': '1',
            'max': '120',
            'step': '1',
            'id': 'id_Age',
            'required': 'required'
        }),
        help_text='Patient age in completed years (1–120).'
    )

    Gender = forms.ChoiceField(
        label='Gender (Biological Sex)',
        choices=GENDER_CHOICES,
        initial='Male',
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_Gender',
            'required': 'required'
        }),
        help_text='Biological sex at birth (Male = 1, Female = 0).'
    )

    # Section 2: Bilirubin Measurements
    Total_Bilirubin = forms.FloatField(
        label='Total Bilirubin',
        min_value=0.01,
        max_value=100.0,
        initial=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 1.0',
            'min': '0.01',
            'max': '100.0',
            'step': '0.1',
            'id': 'id_Total_Bilirubin',
            'required': 'required'
        }),
        help_text='Total blood serum bilirubin in mg/dL (Normal: 0.1–1.2 mg/dL).'
    )

    Direct_Bilirubin = forms.FloatField(
        label='Direct Bilirubin (Conjugated)',
        min_value=0.0,
        max_value=50.0,
        initial=0.2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 0.2',
            'min': '0.0',
            'max': '50.0',
            'step': '0.1',
            'id': 'id_Direct_Bilirubin',
            'required': 'required'
        }),
        help_text='Conjugated (direct) bilirubin in mg/dL (Normal: 0.0–0.3 mg/dL).'
    )

    # Section 3: Liver Enzyme Measurements
    Alkaline_Phosphotase = forms.FloatField(
        label='Alkaline Phosphatase (ALP)',
        min_value=1.0,
        max_value=3000.0,
        initial=190.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 190',
            'min': '1',
            'max': '3000',
            'step': '1',
            'id': 'id_Alkaline_Phosphotase',
            'required': 'required'
        }),
        help_text='Serum alkaline phosphatase enzyme activity in IU/L (Normal: 44–147 IU/L).'
    )

    Alamine_Aminotransferase = forms.FloatField(
        label='Alamine Aminotransferase (ALT / SGPT)',
        min_value=1.0,
        max_value=3000.0,
        initial=25.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 25',
            'min': '1',
            'max': '3000',
            'step': '1',
            'id': 'id_Alamine_Aminotransferase',
            'required': 'required'
        }),
        help_text='Serum alanine aminotransferase enzyme activity in IU/L (Normal: 7–56 IU/L).'
    )

    Aspartate_Aminotransferase = forms.FloatField(
        label='Aspartate Aminotransferase (AST / SGOT)',
        min_value=1.0,
        max_value=5000.0,
        initial=30.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 30',
            'min': '1',
            'max': '5000',
            'step': '1',
            'id': 'id_Aspartate_Aminotransferase',
            'required': 'required'
        }),
        help_text='Serum aspartate aminotransferase enzyme activity in IU/L (Normal: 10–40 IU/L).'
    )

    # Section 4: Protein Measurements
    Total_Proteins = forms.FloatField(
        label='Total Proteins',
        min_value=1.0,
        max_value=15.0,
        initial=6.8,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 6.8',
            'min': '1.0',
            'max': '15.0',
            'step': '0.1',
            'id': 'id_Total_Proteins',
            'required': 'required'
        }),
        help_text='Total serum protein concentration in g/dL (Normal: 6.0–8.3 g/dL).'
    )

    Albumin = forms.FloatField(
        label='Albumin',
        min_value=0.5,
        max_value=10.0,
        initial=3.2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 3.2',
            'min': '0.5',
            'max': '10.0',
            'step': '0.1',
            'id': 'id_Albumin',
            'required': 'required'
        }),
        help_text='Serum albumin concentration in g/dL (Normal: 3.4–5.4 g/dL).'
    )

    Albumin_and_Globulin_Ratio = forms.FloatField(
        label='Albumin and Globulin Ratio (A/G Ratio)',
        min_value=0.1,
        max_value=5.0,
        initial=0.9,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 0.9',
            'min': '0.1',
            'max': '5.0',
            'step': '0.05',
            'id': 'id_Albumin_and_Globulin_Ratio',
            'required': 'required'
        }),
        help_text='Ratio of albumin to globulins (Normal: 0.8–2.0).'
    )

    def clean(self):
        cleaned_data = super().clean()

        # Check for NaN or infinite values across all numeric fields
        for field_name, value in cleaned_data.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    self.add_error(field_name, "Invalid numeric value (NaN or infinity detected).")
                elif value < 0:
                    self.add_error(field_name, "Clinical laboratory values cannot be negative.")

        tb = cleaned_data.get('Total_Bilirubin')
        db = cleaned_data.get('Direct_Bilirubin')
        if tb is not None and db is not None:
            if db > tb:
                # Add validation error or warning: Direct bilirubin cannot exceed total bilirubin
                self.add_error('Direct_Bilirubin', "Direct bilirubin cannot be greater than Total Bilirubin.")

        alb = cleaned_data.get('Albumin')
        tp = cleaned_data.get('Total_Proteins')
        if alb is not None and tp is not None:
            if alb > tp:
                self.add_error('Albumin', "Albumin cannot exceed Total Proteins concentration.")

        return cleaned_data
