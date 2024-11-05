import time
from typing import Optional
import odrive
from odrive import enums as oenum
import numpy as np
import tkinter as tk

import odriveapi

USE_ROBOT = True


AXIS_0_MIN_POS = 0
AXIS_0_MAX_POS = 10
AXIS_1_MIN_POS = 0
AXIS_1_MAX_POS = 10

thresh = 0.008

def handle_press(event):
    global mouse_pressed
    mouse_pressed = True


def handle_release(event):
    global mouse_pressed
    mouse_pressed = False
    
def calcul_draw(odrv, canvas, event):
    global coords

    if not mouse_pressed:
        canvas.old_coords = None
        return

    # Récupérer les coordonnées actuelles de la souris
    y, x  = event.x / 40, event.y / 40  # Mise à l'échelle pour correspondre aux limites 0-10

    # Appliquer les limites pour les positions
    x = max(AXIS_0_MIN_POS, min(x,AXIS_0_MAX_POS))  # Limiter X entre x_min et x_max
    y = max(AXIS_1_MIN_POS, min(y, AXIS_1_MAX_POS)) # Limiter Y entre y_min et y_max
    
    if canvas.old_coords is not None:
        # Calculer les positions cibles pour ODrive
        odrv.cartesian_move(x,y, vel=2)
        
        tab = odrv.curr_xy
        # Attendre que chaque axe atteigne la position cible
        while (abs(tab[0] - x) > thresh or 
               abs(tab[1] - y) > thresh):
            time.sleep(0.001)  # Pause pour éviter un CPU à 100%
    canvas.old_coords = x, y
    
def draw(odrv):
    # Configurer l'interface graphique pour tracer les mouvements
    root = tk.Tk()
    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()
    canvas.old_coords = None

    # Associer les événements de la souris
    root.bind("<Motion>", lambda event: calcul_draw(odrv, canvas, event))
    root.bind("<ButtonPress-1>", handle_press)
    root.bind("<ButtonRelease-1>", handle_release)
    root.mainloop()


def circle(odrv):
    resolution = 10

    x1 = np.linspace(0, 1, resolution)
    y1 = np.sqrt(np.ones_like(x1) - np.power(x1, 2 * np.ones_like(x1)))
    x = np.append(x1, [np.flip(x1), -x1, -np.flip(x1)])
    y = np.append(-y1, [np.flip(y1), y1, -np.flip(y1)])

    x = (x * 2) + 5
    y = (y * 2) + 5

    # print(x)
    # print(y)

    # ax = plt.figure().add_subplot()
    # ax.plot(x, y)
    # plt.show()

    thresh = 0.008
    for i, _ in enumerate(x):
        
        odrv.cartesian_move(x[i],y[i], vel=2)
        tab = odrv.curr_xy

        while abs(tab[0] - x[i]) > thresh and  abs(tab[1] - y[i]) > thresh:
            time.sleep(0.001)
            
def cartesien(odrv, x, y, vel=2):
    odrv.cartesian_move(x,y, vel)
            

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
    
    #circle(odrv)
    cartesien(odrv,10,10)
    #draw(odrv)


    odrv.shutdown()

    return 0


main()
