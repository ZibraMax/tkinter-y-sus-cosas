import math
from tkinter import Tk, Canvas, W
import time

# Definición recursiva, requiere numpy
# def B(t,P):
# 	if len(P)==1:
# 		return np.array(P[0])
# 	else:
# 		return (1-t)*B(t,P[:-1])+t*B(t,P[1:])


def binomial(i, n):
    return math.factorial(n) / float(
        math.factorial(i) * math.factorial(n - i))


def bernstein(t, i, n):
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(t, points):
    n = len(points) - 1
    x = y = 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    return x, y


def drawBezier():
    global editando, animacion
    my_canvas.create_text(100, height-150, fill="black", font='20',
                          text="Click Izquierdo: Agregar Punto", anchor=W)

    my_canvas.create_text(100, height-130, fill="black", font='20',
                          text="Click Derecho: Mover punto", anchor=W)

    my_canvas.create_text(100, height-110, fill="black", font='20',
                          text='Tecla "D": Reset', anchor=W)
    my_canvas.create_text(100, height-90, fill="black", font='20',
                          text='Tecla "A": Animación aal agregar punto', anchor=W)
    PS = []
    r = 3
    M = int(50*len(P))
    for t in range(M+1):
        PS += [bezier(t/M, P)]

    for i in range(len(P)-1):
        x = P[i][0]
        y = P[i][1]
        my_canvas.create_line(
            x, y, P[i+1][0], P[i+1][1], fill='gray', width=1, dash=(3, 3))
        my_canvas.create_oval(x-r, y-r, x+r, y+r, fill="blue")
    x = P[-1][0]
    y = P[-1][1]
    my_canvas.create_oval(x-r, y-r, x+r, y+r, fill="blue")
    for i in range(len(PS)-1):
        x1 = PS[i][0]
        y1 = PS[i][1]
        x2 = PS[i+1][0]
        y2 = PS[i+1][1]
        my_canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
        if not editando and animacion:
            time.sleep((1*10**(-15))/M)
            my_canvas.update()


def addPoint(event):
    my_canvas.delete("all")
    global P
    x = event.x
    y = event.y
    P += [[x, y]]
    drawBezier()


def editPoint(event):
    global P
    global editando
    global masCercano
    if len(P) > 1:
        editando = not editando
        x = event.x
        y = event.y
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
        x = event.x
        y = event.y
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
