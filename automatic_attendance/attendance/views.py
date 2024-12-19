from django.shortcuts import render, redirect, get_object_or_404
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from django.conf import settings
from sklearn.neighbors import KNeighborsClassifier
import joblib
from .models import User, Attendance
from .forms import UserForm
from django.contrib import messages


# Chemin vers le fichier haarcascade
face_cascade_path = os.path.join(
    settings.BASE_DIR, "static/haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(face_cascade_path)

# Vérifiez que le détecteur est chargé avec succès
if face_detector.empty():
    raise Exception("Échec du chargement du fichier XML Haar Cascade.")

# Répertoire où stocker les fichiers CSV
BASE_CSV_DIR = "attendance_records/"


def initialize_user_monthly_csv(user):
    current_month = datetime.now().strftime("%Y-%m")
    user_dir = os.path.join(BASE_CSV_DIR, user.ID_number)

    # Créer le répertoire de l'utilisateur s'il n'existe pas
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    csv_file_path = os.path.join(user_dir, f"{current_month}.csv")

    # Si le fichier CSV n'existe pas, créez-le avec les en-têtes
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, "w") as f:
            f.write(
                "Username,ID_number,Date,Time\n"
            ) 
            print(f"Fichier CSV créé : {csv_file_path}")

    return csv_file_path


def add_attendance(user):
    csv_file_path = initialize_user_monthly_csv(
        user
    )  # Appel à la fonction d'initialisation
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Charger le fichier CSV dans un DataFrame pandas
    df = pd.read_csv(csv_file_path)

    # Si l'utilisateur n'est pas déjà présent, ajoutez-le
    if user.ID_number not in df["ID_number"].values:
        with open(csv_file_path, "a") as f:
            f.write(
                f"{user.username},{user.ID_number},{current_date},{current_time}\n"
            )
            print(f"Données ajoutées pour l'utilisateur : {user.username}")

        # Enregistrer dans le modèle Attendance
        attendance_record = Attendance(user=user)  # Créez un enregistrement d'assiduité
        attendance_record.save()  # Enregistrez-le dans la base de données


def totalreg():
    return User.objects.count()


def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        print(f"Visages détectés : {len(face_points)}")  # Ajouté pour le débogage
        return face_points
    return []


def identify_face(facearray):
    try:
        model = joblib.load("static/face_recognition_model.pkl")
        return model.predict(facearray)
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle : {e}")


def train_model():
    faces = []
    labels = []
    users = User.objects.all()
    for user in users:
        user_images_path = os.path.join(
            "static/faces", f"{user.username}_{user.ID_number}"
        )
        for imgname in os.listdir(user_images_path):
            img = cv2.imread(os.path.join(user_images_path, imgname))
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user.username)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")


def home(request):
    attendance_records = Attendance.objects.all()
    return render(
        request,
        "attendance/home.html",
        {
            "attendance_records": attendance_records,
            "totalreg": totalreg(),
        },
    )


def start_attendance(request):
    # Vérifiez si le modèle de reconnaissance faciale existe
    if not os.path.isfile("static/face_recognition_model.pkl"):
        return render(
            request,
            "attendance/home.html",
            {
                "totalreg": totalreg(),
                "mess": "Aucun modèle entraîné trouvé. Veuillez ajouter un nouvel utilisateur.",
            },
        )

    # Initialisez la webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Lire une image de la webcam
        if not ret:
            print("Erreur de lecture de la webcam.")
            break

        # Extraire les visages détectés
        faces_detected = extract_faces(frame)
        print(f"Nombre de visages détectés : {len(faces_detected)}")  # Débogage

        # Parcourir tous les visages détectés
        for (x, y, w, h) in faces_detected:
            # Dessiner un rectangle autour du visage détecté
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (0, 0, 255), -1)

            # Préparez le visage pour l'identification
            face = cv2.resize(frame[y: y + h, x: x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            print(f"Personne identifiée : {identified_person}")

            # Récupérer l'utilisateur identifié
            user = User.objects.filter(username=identified_person).first()

            if user:
                # Obtenir la date d'aujourd'hui
                today = datetime.now().date()

                # Vérifiez si l'utilisateur a déjà une présence enregistrée pour aujourd'hui
                attendance_today = Attendance.objects.filter(
                    user=user,
                    timestamp__date=today
                ).exists()

                if not attendance_today:
                    # Ajoutez la présence de l'utilisateur
                    add_attendance(user)
                    print(f"Présence ajoutée pour {identified_person}")

                else:
                    print(f"Présence déjà enregistrée pour {identified_person}")

            # Afficher le nom de la personne sur l'image
            cv2.putText(
                frame,
                f"{identified_person}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Afficher l'image avec les visages détectés et le texte
        cv2.imshow("Attendance", frame)

        # Quitter si la touche Échap est pressée
        if cv2.waitKey(1) & 0xFF == 27:  # Échap
            break

    # Libérer la webcam et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()

    # Récupérer tous les enregistrements de présences
    attendance_records = Attendance.objects.all()
    print(f"Total attendance records: {attendance_records.count()}")  # Débogage

    # Renvoyer la vue avec les enregistrements
    return render(
        request,
        "attendance/home.html",
        {
            "attendance_records": attendance_records,
            "totalreg": totalreg(),
        },
    )



def add_user(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            userimagefolder = os.path.join("static/faces", f"{user.username}_{user.ID_number}")
            os.makedirs(userimagefolder, exist_ok=True)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messages.error(request, "Impossible d'accéder à la webcam.")
                return render(request, "attendance/add_user.html", {"form": form})

            messages.info(request, 'Ajustez votre position devant la webcam. Appuyez sur "E" pour capturer l\'image.')

            images_captured = 0  # Compteur pour les images capturées

            while images_captured < 2:  # Continue jusqu'à ce que 2 images soient capturées
                ret, frame = cap.read()
                if not ret:
                    messages.error(request, "Erreur de capture vidéo.")
                    break

                cv2.imshow("Ajustez votre position", frame)

                key = cv2.waitKey(1)
                if key == 27:  # Échap
                    break
                elif key == ord("e"):  # Touche E pour capturer l'image
                    # Enregistrement de l'image capturée
                    image_path = os.path.join(userimagefolder, f"{user.username}_captured_{images_captured + 1}.jpg")
                    cv2.imwrite(image_path, frame)
                    messages.success(request, f"Image {images_captured + 1} capturée avec succès !")
                    images_captured += 1  # Incrémente le compteur d'images capturées

            cap.release()
            cv2.destroyAllWindows()
           

            return render(request, "attendance/close_tab.html", {
                "username": user.username,
                "ID_number": user.ID_number,
                "message": "Enregistrement réussi !",
            })
        else:
            messages.error(request, "Erreur dans le formulaire. Veuillez réessayer.")
            return render(request, "attendance/add_user.html", {"form": form})

    else:
        form = UserForm()

    return render(request, "attendance/add_user.html", {"form": form})


def delete_all_attendance(request):
    Attendance.objects.all().delete()
    messages.success(request, 'Tous les enregistrements d\'assistance ont été supprimés avec succès !')
    return redirect('home')
