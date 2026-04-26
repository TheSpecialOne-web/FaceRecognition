# Face Recognition Attendance System

A Django web app that tracks attendance via **webcam-based face recognition**.

## Features

- 👤 User registration with multi-image capture via webcam
- 🎯 Face detection using OpenCV Haar Cascade
- 🤖 Face recognition with **scikit-learn KNN classifier**
- 📅 Monthly attendance logs (CSV per user, per month)
- 🗄️ Django admin panel for managing users and records

## Architecture

```
automatic_attendance/
├── attendance/              # Main Django app
│   ├── models.py            # User, Attendance models
│   ├── views.py             # Detection, registration, attendance flow
│   ├── forms.py
│   └── urls.py
├── automatic_attendance/    # Project config
│   └── settings.py
├── static/
│   └── haarcascade_frontalface_default.xml
├── manage.py
└── requirements.txt
```

### Recognition pipeline

1. Capture face from webcam (Haar Cascade detection)
2. Convert to grayscale, resize to fixed dimensions
3. Train a KNN classifier on stored user faces (re-trained on registration)
4. At attendance time, predict the closest match
5. Log timestamped entry into the user's monthly CSV (auto-created)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r automatic_attendance/requirements.txt

cd automatic_attendance
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open http://localhost:8000

## Stack

- Python 3.10+
- Django 4.x
- OpenCV (`cv2`)
- scikit-learn (KNN)
- Pandas
- Bootstrap (frontend templates)

## License

MIT
