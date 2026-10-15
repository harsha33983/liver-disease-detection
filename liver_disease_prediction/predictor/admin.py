from django.contrib import admin
from .models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'assessment_id',
        'created_at',
        'age',
        'gender',
        'predicted_class',
        'confidence_display',
        'created_by'
    )
    list_filter = ('predicted_class', 'gender', 'created_at')
    search_fields = ('assessment_id',)
    readonly_fields = (
        'assessment_id',
        'created_at',
        'predicted_class',
        'probability_disease',
        'probability_no_disease',
        'model_version'
    )
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
