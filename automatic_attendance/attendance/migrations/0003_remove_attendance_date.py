# Generated by Django 5.1.1 on 2024-10-09 02:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("attendance", "0002_attendance_date"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="attendance",
            name="date",
        ),
    ]