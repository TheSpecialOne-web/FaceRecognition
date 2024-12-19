from django import forms
from .models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'ID_number']
        widgets = {
            'username': forms.TextInput(attrs={'placeholder': 'Enter Username'}),
            'ID_number': forms.TextInput(attrs={'placeholder': 'Enter Roll Number'}),
        }
