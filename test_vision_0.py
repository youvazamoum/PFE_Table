import logging
import cv2
import numpy as np
import mediapipe as mp
from numpy.typing import NDArray
from typing import Tuple

# Configure le logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__file__)

# Initialise MediaPipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Pour dessiner les points clés

# Crée une figure de 10x10 pour le dessin
drawing_canvas = np.zeros((10, 10), dtype=np.uint8)

def cap() -> Tuple[cv2.VideoCapture, NDArray[np.int32]]:
    try:
        # Utilise la webcam intégrée avec l'index 0
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise Exception("Impossible d'ouvrir la webcam intégrée.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise Exception("Impossible de lire une image depuis la webcam.")
    except Exception:
        logger.error("Erreur lors de la configuration de la capture vidéo.", exc_info=True)
        cap.release()
        raise SystemExit(1)

    return cap, np.array(frame.shape, dtype=np.int32)

# Initialisation de la capture et récupération de la forme de l'image
cap, frame_shape = cap()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de l'image.")
            break

        # Réinitialise le canevas à chaque itération pour effacer les anciens dessins
        drawing_canvas.fill(0)

        # Conversion en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection de la main avec MediaPipe
        results = hands.process(rgb_frame)

        # Si une main est détectée
        if results.multi_hand_landmarks:
            # Récupère le premier point clé de l'index (landmark 8)
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convertit les coordonnées normalisées de MediaPipe en pixels
            h, w, _ = frame.shape
            x_pixel = int(index_tip.x * w)
            y_pixel = int(index_tip.y * h)

            # Mise à l'échelle des coordonnées de l'index pour le canvas de 10x10
            x_scaled = min(9, max(0, int(x_pixel / w * 10)))
            y_scaled = min(9, max(0, int(y_pixel / h * 10)))

            # Trace sur le dessin avec la position de l'index
            drawing_canvas[y_scaled, x_scaled] = 255  # Valeur blanche pour marquer le dessin

        # Agrandit le canvas pour l'affichage
        display_canvas = cv2.resize(drawing_canvas, (200, 200), interpolation=cv2.INTER_NEAREST)

        # Affiche les points clés de la main sur la fenêtre principale
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Affiche les deux fenêtres
        cv2.imshow("Webcam - Détection de la main", frame)
        cv2.imshow("Dessin", display_canvas)

        # Appuyez sur 'q' pour quitter la fenêtre
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Libère la capture et ferme toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()
    hands.close()  # Ferme l'objet de détection des mains de MediaPipe
