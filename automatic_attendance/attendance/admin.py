from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import User, Attendance

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'ID_number')

