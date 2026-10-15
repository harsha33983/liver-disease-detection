from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('result/<str:assessment_id>/', views.prediction_result_view, name='prediction_result'),
    path('history/', views.prediction_history_view, name='prediction_history'),
    path('history/<str:assessment_id>/', views.prediction_detail_view, name='prediction_detail'),
    path('model-info/', views.model_info_view, name='model_info'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
