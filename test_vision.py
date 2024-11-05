import logging
import cv2
import numpy as np
import mediapipe as mp
from numpy.typing import NDArray
from typing import Tuple, Optional
import time
import odrive
from odrive import enums as oenum
import threading
import math
import odriveapi

USE_ROBOT = True


AXIS_0_MIN_POS = 0
AXIS_0_MAX_POS = 10
AXIS_1_MIN_POS = 0
AXIS_1_MAX_POS = 10

thresh = 0.001


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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Impossible d'ouvrir la webcam intégrée.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise Exception("Impossible de lire une image depuis la webcam.")
    except Exception:
        logger.error("Erreur lors de la configuration de la capture vidéo.", exc_info=True)
        cap.release()
        raise SystemExit(1)

    return cap, np.array(frame.shape, dtype=np.int32)

# Variable pour stocker le temps de la dernière commande envoyée
last_x, last_y = None, None  # Dernières positions X et Y du robot
movement_threshold = 1

def vis(odrv):
    global cap , last_x ,last_y
    # Initialisation de la capture et récupération de la forme de l'image
    cap, frame_shape = cap()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture de l'image.")
                break
    
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
                #x_scaled = min(9, max(0, int(x_pixel / w * 10)))
                #y_scaled = min(9, max(0, int(y_pixel / h * 10)))
                #print("x =", x_scaled,"y =", y_scaled)
                
                # Mise à l'échelle des coordonnées et adaptation au repère du robot
                y_scaled = min(9, max(0, int(x_pixel / w * 10)))      # x_scaled devient y dans le repère du robot
                x_scaled = min(9, max(0, int((1 - y_pixel / h) * 10)))     # y_scaled devient -x dans le repère du robot
                print("x_robot =", x_scaled, "y_robot =", y_scaled)

    
                # Trace sur le dessin avec la position de l'index
                drawing_canvas[y_scaled, x_scaled] = 255  # Valeur blanche pour marquer le dessin
                
                # Appliquer les limites pour les positions
                #x_scaled = max(AXIS_0_MIN_POS, min(x,AXIS_0_MAX_POS))  # Limiter X entre x_min et x_max
                #y_scaled = max(AXIS_1_MIN_POS, min(y, AXIS_1_MAX_POS)) # Limiter Y entre y_min et y_max

                if last_x is None or abs(x_scaled - last_x) > movement_threshold or abs(y_scaled - last_y) > movement_threshold:
                    last_x, last_y = x_scaled, y_scaled
                         
                # Lancer le mouvement des moteurs dans un autre thread
                    threading.Thread(target=move, args=(x_scaled, y_scaled, odrv)).start()
                """
                # Calculer les positions cibles pour ODrive
                odrv.cartesian_move(x_scaled,y_scaled, vel=2)
                    
                tab = odrv.curr_xy
                # Attendre que chaque axe atteigne la position cible
                while (abs(tab[0] - x_scaled) > thresh or 
                       abs(tab[1] - y_scaled) > thresh):
                    time.sleep(0.001)  # Pause pour éviter un CPU à 100%
                """
                              
            # Agrandit le canvas pour l'affichage
            display_canvas = cv2.resize(drawing_canvas, (200, 200), interpolation=cv2.INTER_NEAREST)
    
            # Affiche les points clés de la main sur la fenêtre principale
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
            # Affiche les deux fenêtres
            cv2.imshow("Webcam - Detection de la main", frame)
            cv2.imshow("Dessin", display_canvas)
    
            # Appuyez sur 'q' pour quitter la fenêtre
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Libère la capture et ferme toutes les fenêtres
        cap.release()
        cv2.destroyAllWindows()
        hands.close()  # Ferme l'objet de détection des mains de MediaPipe
        
def move(x_scaled, y_scaled, odrv):
    """Contrôle de la vitesse en fonction de la distance pour le déplacement en position."""
    # Obtenir la position actuelle
    tab = odrv.curr_xy
    current_x, current_y = tab[0], tab[1]

    # Calculer la distance vers la nouvelle position
    distance = math.sqrt((x_scaled - current_x) ** 2 + (y_scaled - current_y) ** 2)

    # Définir une vitesse proportionnelle à la distance
    min_speed = 0.5
    max_speed = 5.0
    speed = min_speed + (max_speed - min_speed) * (distance / 10)
    speed = min(max_speed, max(min_speed, speed))

    # Utiliser `cartesian_move` avec la vitesse calculée et sans boucle d'attente
    odrv.cartesian_move(x_scaled, y_scaled, vel=speed)
    
    
    # """Thread de mouvement des moteurs ODrive."""
    # odrv.cartesian_move(x_scaled, y_scaled, vel=2)
    # tab = odrv.curr_xy
    
    # # Attendre que chaque axe atteigne la position cible
    # while (abs(tab[0] - x_scaled) > thresh or 
    #        abs(tab[1] - y_scaled) > thresh):
    #     time.sleep(0.001)  # Pause pour éviter un CPU à 100%

def main(
    odrv: Optional[odriveapi.OdriveAPI] = None,
) -> int:
    if odrv is None:
        # find the odrive
        if USE_ROBOT:
            _odrv = odrive.find_any()
            _odrv.axis0.controller.config.vel_limit = 5
            _odrv.axis1.controller.config.vel_limit = 5
            _odrv.axis0.controller.config.vel_limit_tolerance = 500000
            _odrv.axis1.controller.config.vel_limit_tolerance = 500000
        else:
            _odrv = None

        odrv = odriveapi.OdriveAPI(_odrv, USE_ROBOT)
        odrv.startup()

    odrv._odrv.clear_errors()
    time.sleep(0.1)
    odrv._odrv.clear_errors()
    time.sleep(0.1)

    # Exemple de fonction 
    vis(odrv)


    odrv.shutdown()

    return 0


main()