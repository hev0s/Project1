import dlib
import os

path = "models/shape_predictor_68_face_landmarks.dat"

print("Chemin absolu attendu :", os.path.abspath(path))

if not os.path.exists(path):
    raise FileNotFoundError("⚠️ Le fichier du modèle Dlib est introuvable !")

print("✅ Modèle Dlib trouvé avec succès.")
