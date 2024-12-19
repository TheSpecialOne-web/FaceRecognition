from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Main page
    path('start/', views.start_attendance, name='start_attendance'),  # Start attendance
    path('add/', views.add_user, name='add_user'),  # Add a new user
    path('attendance/', views.add_attendance, name='attendance'),  # View attendance records
    path('delete_all_attendance/', views.delete_all_attendance, name='delete_all_attendance'),
    
]

