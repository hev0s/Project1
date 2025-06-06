import cv2
import socket
import numpy as np
import struct

# Configuration
HOST = '127.0.0.1'      # Adresse IP du serveur (DeepFaceLive en écoute)
PORT = 9999             # Port utilisé par DeepFaceLive ou ton proxy serveur

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

# Crée un socket client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('wb')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionne l'image si nécessaire
        frame = cv2.resize(frame, (640, 480))

        # Encode en JPEG
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        data = jpeg_frame.tobytes()

        # Envoie la taille de l'image (pour que le serveur sache quoi lire)
        client_socket.sendall(struct.pack('<L', len(data)))
        client_socket.sendall(data)

        # Affichage local pour contrôle
        cv2.imshow('Sending to DeepFaceLive', frame)
        if cv2.waitKey(1) == 27:  # ESC pour quitter
            break
finally:
    cap.release()
    connection.close()
    client_socket.close()
    cv2.destroyAllWindows()


"""
OBS 

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam to OBS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""