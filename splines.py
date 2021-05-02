import math
from tkinter import Tk, Canvas, W
import time
import numpy as np


def splines(P):
    X = np.array(P)[:, 0]
    Y = np.array(P)[:, 1]
    n = len(P) - 1
    K = np.zeros([3*n, 3*n])
    F = np.zeros([3*n, 1])
    count = 0
    K = np.zeros([3*n, 3*n])
    F = np.zeros([3*n, 1])
    for i in range(0, 3*n, 3):
        x0 = X[count]
        y0 = Y[count]
        x1 = X[count+1]
        y1 = Y[count+1]

        F[i, 0] = y0
        F[i+1, 0] = y1

        K[i, i] = x0**2
        K[i, i+1] = x0
        K[i, i+2] = 1

        K[i+1, i] = x1**2
        K[i+1, i+1] = x1
        K[i+1, i+2] = 1

        K[i+2, i] = 2*x1
        K[i+2, i+1] = 1
        if (i+3) <= (3*n-1):
            K[i+2, i+3] = -2*x1
            K[i+2, i+4] = -1

        count += 1
    try:
        K[-1] = 0
        K[-1, 0] = 1
        F[-1, 0] = 0
        U = np.linalg.solve(K, F)
        return U
    except Exception as e:
        pass


def splinesO3(P):
    n = len(P) - 1
    X = np.array(P)[:, 0]
    Y = np.array(P)[:, 1]
    count = 0
    K = np.zeros([4*n, 4*n])
    F = np.zeros([4*n, 1])
    for i in range(0, 4*n, 4):
        x0 = X[count]
        y0 = Y[count]
        x1 = X[count+1]
        y1 = Y[count+1]
        count += 1
        F[i, 0] = y0
        F[i+1, 0] = y1

        K[i, i] = x0**3
        K[i, i+1] = x0**2
        K[i, i+2] = x0
        K[i, i+3] = 1

        K[i+1, i] = x1**3
        K[i+1, i+1] = x1**2
        K[i+1, i+2] = x1
        K[i+1, i+3] = 1

        if (i+4) <= (4*n-1):
            K[i+2, i] = 3*x1**2
            K[i+2, i+1] = 2*x1
            K[i+2, i+2] = 1

            K[i+2, i+4] = -3*x1**2
            K[i+2, i+5] = -2*x1
            K[i+2, i+6] = -1

            K[i+3, i] = 6*x1
            K[i+3, i+1] = 2

            K[i+3, i+4] = -6*x1
            K[i+3, i+5] = -2

    try:
        K[-2, 0] = 6*X[0]
        K[-2, 1] = 2

        K[-1, -4] = 6*X[-1]
        K[-1, -3] = 2
        F[-1, 0] = 0
        F[-2, 0] = 0
        return np.linalg.solve(K, F)
    except Exception as e:
        pass


def drawBezier():
    global editando, animacion
    my_canvas.create_text(100, 20, fill="red", font='20',
                          text="Splines orden 3", anchor=W)
    my_canvas.create_text(100, 50, fill="blue",
                          font='20', text="Splines orden 2", anchor=W)
    my_canvas.create_text(100, height-150, fill="black", font='20',
                          text="Click Izquierdo: Agregar Punto", anchor=W)

    my_canvas.create_text(100, height-130, fill="black", font='20',
                          text="Click Derecho: Mover punto", anchor=W)

    my_canvas.create_text(100, height-110, fill="black", font='20',
                          text='Tecla "D": Reset', anchor=W)
    my_canvas.create_text(100, height-90, fill="black", font='20',
                          text='Tecla "A": Animación aal agregar punto', anchor=W)
    U = splines(P)
    UO3 = splinesO3(P)
    PS = []
    PSO3 = []
    r = 3
    M = 30
    n = len(P) - 1
    for i in range(n):

        _x = np.linspace(P[i][0], P[i+1][0], M)
        _y = U[3*i, 0]*_x**2+U[3*i+1, 0]*_x+U[3*i+2, 0]
        _y2 = UO3[4*i, 0]*_x**3+UO3[4*i+1, 0] * \
            _x**2 + UO3[4*i+2, 0]*_x+UO3[4*i+3, 0]
        PS += np.array([_x, _y]).T.tolist()
        PSO3 += np.array([_x, _y2]).T.tolist()
    for i in range(len(P)-1):
        x = P[i][0]
        y = P[i][1]
        my_canvas.create_line(
            x, y, P[i+1][0], P[i+1][1], fill='gray', width=1, dash=(3, 3))
        my_canvas.create_oval(x-r, y-r, x+r, y+r, fill="yellow")
    x = P[-1][0]
    y = P[-1][1]
    my_canvas.create_oval(x-r, y-r, x+r, y+r, fill="yellow")
    for i in range(len(PS)-1):
        x1 = PS[i][0]
        y1 = PS[i][1]
        x2 = PS[i+1][0]
        y2 = PS[i+1][1]
        my_canvas.create_line(x1, y1, x2, y2, fill='blue', width=2)
        my_canvas.create_line(PSO3[i][0], PSO3[i][1], PSO3[i+1]
                              [0], PSO3[i+1][1], fill='red', width=2)
        if not editando and animacion:
            time.sleep((1*10**(-15))/M)
            my_canvas.update()


def addPoint(event):
    my_canvas.delete("all")
    global P
    x = event.x
    y = event.y
    P += [[float(x), float(y)]]
    drawBezier()


def editPoint(event):
    global P
    global editando
    global masCercano
    if len(P) > 1:
        editando = not editando
        x = float(event.x)
        y = float(event.y)
        minima = float("inf")
        for i, p in enumerate(P):
            dist = math.sqrt((p[0]-x)**2+(p[1]-y)**2)
            if dist < minima:
                minima = dist
                masCercano = i


def movePoint(event):
    global editando
    if editando:
        my_canvas.delete("all")
        global P
        global masCercano
        x = float(event.x)
        y = float(event.y)
        P[masCercano][0] = x
        P[masCercano][1] = y
        drawBezier()


def kpup(e):
    global P
    global editando, animacion
    if e.char == 'd':
        P = []
        editando = False
        my_canvas.delete("all")
    if e.char == 'a':
        animacion = not animacion


my_window = Tk()
width = my_window.winfo_screenwidth()
height = my_window.winfo_screenheight()
my_canvas = Canvas(my_window, width=width, height=height,
                   background='white', cursor="circle")
my_canvas.grid(row=0, column=0)
my_canvas.bind('<Button-1>', addPoint)
my_canvas.bind('<Button-3>', editPoint)
my_canvas.bind('<Motion>', movePoint)
my_window.bind('<KeyRelease>', kpup)

P = []
masCercano = None
animacion = False
editando = False
my_window.title(
    'Left click: add point. Right click: move nearest point. D: reset')
my_window.state('zoomed')


my_canvas.create_text(100, height-150, fill="black", font='20',
                      text="Click Izquierdo: Agregar Punto", anchor=W)

my_canvas.create_text(100, height-130, fill="black", font='20',
                      text="Click Derecho: Mover punto", anchor=W)

my_canvas.create_text(100, height-110, fill="black", font='20',
                      text='Tecla "D": Reset', anchor=W)
my_canvas.create_text(100, height-90, fill="black", font='20',
                      text='Tecla "A": Animación aal agregar punto', anchor=W)
my_window.mainloop()
