from django.db import models

# Create your models here.
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=150)

    ID_number = models.CharField(max_length=10,unique=True)

    def __str__(self):
        return f'{self.username} ({self.ID_number})'

class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
   

    def __str__(self):
        return f'Attendance for {self.user.username} at {self.timestamp}'

