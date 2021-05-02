import math
from tkinter import Tk, Canvas, W, E, NW
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from scipy.interpolate import interp1d
import time
import numpy as np
# Definición recursiva, requiere numpy
# def B(t,P):
# 	if len(P)==1:
# 		return np.array(P[0])
# 	else:
# 		return (1-t)*B(t,P[:-1])+t*B(t,P[1:])


def getCubic(L, disp):
    K = np.array([[3*L**2, 2*L], [L**3, L**2]])
    F = np.array([[np.arctan2(disp, L)], [disp]])
    return np.linalg.solve(K, F)[:, 0]


def graficarBola():
    my_canvas.delete("all")
    graphAcel()
    ponerTextos()
    global width, height, mult, u, L, m, k, multu
    up = u*mult*multu
    Lp = L*mult
    xi, yi = Lp, 0
    centrox, centroy = width/2, height-100
    r = m*2
    my_canvas.create_line(centrox, centroy, centrox,
                          centroy-Lp+r, fill='gray', width=1, dash=[3, 3])
    my_canvas.create_oval(yi-r+centrox, centroy-(xi-r), yi +
                          r+centrox, centroy-(xi+r), fill="")
    theta = np.arctan2(up, Lp)
    Lp = Lp*np.cos(theta)
    up = up*np.cos(theta)
    a, b = getCubic(Lp, up)
    x, y = 0, 0
    for i in range(51):
        equis = Lp/50*i
        xi, yi = equis, a*equis**3+b*equis**2
        my_canvas.create_line(y+centrox, centroy-x, yi+centrox,
                              centroy-xi, fill='red', width=max(1, int(k/100)))
        x, y = xi, yi
    my_canvas.create_oval(y-r+centrox, centroy-(x-r), y +
                          r+centrox, centroy-(x+r), fill="blue")


def editPoint(event):
    global editando
    editando = not editando
    if not editando:
        drawBezier()


def drawBezier():
    global editando, u, v, L, z, f, m, k, dt, t, height, width, T, U, V
    U = []
    T = []
    c = 2*z*m*np.sqrt(k/m)
    while not editando:
        a = -m*f(t)*9.81
        k1u = v
        k1v = -1/m*(c*v+k*u+a)

        ui0 = u + k1u*dt*0.5
        vi0 = v + k1v*dt*0.5

        k2u = vi0
        k2v = -1/m*(c*vi0+k*ui0+a)

        ui1 = u + k2u*dt*0.5
        vi1 = v + k2v*dt*0.5

        k3u = vi1
        k3v = -1/m*(c*vi1+k*ui1+a)

        ui2 = u + k3u*dt
        vi2 = v + k3v*dt

        k4u = vi2
        k4v = -1/m*(c*vi2+k*ui2+a)

        phiu = (k1u+2*k2u+2*k3u+k4u)/6
        phiv = (k1v+2*k2v+2*k3v+k4v)/6

        u = u + phiu*dt
        v = v + phiv*dt
        t += dt
        U += [u]
        V += [v]
        T += [t]
        x0 = 100
        y0 = height-90
        b = 300
        h = 100
        # time.sleep(dt/10)
        graficarBola()
        createGraph(width-b-x0, y0-130, b, h, T, U, title='u [m]', alert=True)
        createGraph(width-b-x0, y0-2*130, b, h, T,
                    V, title='v [m/s]', alert=True)
        my_canvas.update()


def movePoint(event):
    global editando, u, L, width, height
    centrox, centroy = width/2, height-100
    if editando:
        my_canvas.delete("all")
        x = centrox-event.x
        y = centroy-event.y
        P[0] = x
        P[1] = y
        u, L = -x/mult/multu, y/mult
        graficarBola()


def ponerTextos():
    global z, k, m, dt, height, multu, strt
    my_canvas.create_text(100, height-90, fill="black",
                          font='20', text=f"omega={format(np.sqrt(k/m),'.2f')}", anchor=W)
    my_canvas.create_text(200, height-90, fill="black",
                          font='20', text=f"T={format(2*np.pi/np.sqrt(k/m),'.2f')}", anchor=W)
    my_canvas.create_text(100, height-110, fill="black",
                          font='20', text=f"multu={format(multu,'.2f')}", anchor=W)
    my_canvas.create_text(100, height-130, fill="black",
                          font='20', text=f"z={format(z,'.2f')}", anchor=W)
    my_canvas.create_text(100, height-150, fill="black",
                          font='20', text=f"k={format(k,'.2f')}", anchor=W)
    my_canvas.create_text(100, height-170, fill="black",
                          font='20', text=f"m={format(m,'.2f')}", anchor=W)
    my_canvas.create_text(100, height-190, fill="black",
                          font='20', text=f"dt={format(dt,'.2f')}", anchor=W)

    my_canvas.create_text(100, 50, fill="black",
                          font='20', text=strt, anchor=NW)


def createGraph(x0, y0, b, h, X, Y, maxs=None, color='red', title='', alert=False):
    XC = []
    YC = []
    np = 70
    if alert:
        if len(X) > np+1:
            for i in range(0, len(X), int(len(X)/np)):
                XC += [X[i]]
                YC += [Y[i]]
            XC += [X[-1]]
            YC += [Y[-1]]
        else:
            XC = X
            YC = Y
    else:
        XC = X
        YC = Y
    X = XC
    Y = YC
    xf = x0+b
    yf = y0-h
    ym = y0-h/2
    xmax = max(X)
    xmin = min(X)
    if maxs:
        ymax, ymin = maxs
    else:
        ymax = max(Y)
        ymin = min(Y)
    ymax = max(abs(ymax), abs(ymin))
    if ymax == 0:
        ymax = 1
    dx = xmax-xmin
    if dx == 0:
        dx = 1

    def z(x): return (x)/ymax
    X = [(i-xmin)/dx*b for i in X]
    Y = [z(i) for i in Y]
    my_canvas.create_line(x0, y0, x0, yf, fill='gray', width=1)
    my_canvas.create_line(x0, ym, xf, ym, fill='gray', width=1)
    my_canvas.create_text(x0-20, yf-20, fill="black",
                          font='20', text=f"{format(t,'.2f')}", anchor=W)
    my_canvas.create_text(x0-5, ym, fill="black",
                          font='20', text=title, anchor=E)
    for i in range(len(X)-1):
        my_canvas.create_line(x0+X[i], ym-Y[i]*h/2, x0 +
                              X[i+1], ym-Y[i+1]*h/2, fill=color, width=2)


def graphAcel():
    global f, dt, height, t, data, width
    n = 20
    x0 = 100
    y0 = height-90
    b = 300
    h = 100
    dx = b/n
    maxs = None
    X = []
    Y = []
    for i in range(n+1):
        X += [i*dx]
        Y += [f(t+i*dt)]
    try:
        eq = data[:, 0]
        ey = data[:, 1]
        if t < np.max(eq):
            maxs = [np.max(ey), np.min(ey)]
    except:
        pass
    createGraph(width-b-x0, y0, b, h, X, Y, maxs, color='blue', title='a [g]')


def importarArchivo():
    global ARCHIVO
    ARCHIVO = askopenfilename()
    parseArchivo()


def parseArchivo():
    global f, u, v, editando, data
    data = np.loadtxt(ARCHIVO, skiprows=1, delimiter=',')
    f = interp1d(data[:, 0], data[:, 1], kind='linear',
                 fill_value=(0, 0), bounds_error=False)
    u = 0
    v = 0
    graficarBola()
    drawBezier()


def kpup(e):
    global editando, actual, f, u, v, t, U, T
    if e.char.lower() == 'a':
        u, v, t = 0, 0, 0
        def f(x): return 0
        importarArchivo()
    if e.char.lower() == 'r':
        u, v, t = 0, 0, 0
        def f(x): return 0
        U, T = [], []
    if e.char.lower() == 't':
        u, v, t = 0, 0, 0
        U, T = [], []
    else:
        actual = e.char.lower()


def wheel(event):
    global z, k, m, dt, height, actual, multu, editando
    editando = True
    delta = event.delta
    if actual == 'z':
        z += 0.05*np.sign(delta)
        z = max(z, 0)
    elif actual == 'k':
        k += 10*np.sign(delta)
        k = max(k, 0)
    elif actual == 'm':
        m += np.sign(delta)
        m = max(m, 0)
    elif actual == 'd':
        dt += 0.01*np.sign(delta)
        dt = max(dt, 0)
    elif actual == 'u':
        multu += 5*np.sign(delta)
        multu = max(multu, 1)
    graficarBola()
    editando = False
    drawBezier()


my_window = Tk()
ARCHIVO = ''
def f(t): return 0


actual = 'z'
mult = 500
t = 0
u = 0
v = 0
acel = 0
L = 1
z = 0.05
m = 20
k = 1500
dt = 0.01
multu = 100
data = None
U = []
V = []
T = []
ACEL = []
P = [u*mult/multu, L*mult]
strt = "Controles:\nClick: Mover la masa\nA: Seleccionar archivo de aceleración\n\nPara cambiar las propiedades, use una de las siguientes letras\ny cambielas usando la rueda del mouse:\n\nK: Rigidez\nM: Masa\nZ: Amortiguamiento\nd: Paso en el tiempo\nu: Multiplicador de desplazamientos (solo para graficar)\n\nR: Reiniciar todo\nT: Reiniciar tiempo"
width = my_window.winfo_screenwidth()
height = my_window.winfo_screenheight()
my_canvas = Canvas(my_window, width=width, height=height,
                   background='white')
my_canvas.grid(row=0, column=0)
my_canvas.bind('<Button-1>', editPoint)
my_canvas.bind('<Motion>', movePoint)
my_canvas.bind('<MouseWheel>', wheel)
my_window.bind('<KeyRelease>', kpup)
editando = False
my_window.title('Amortiguada')
my_window.state('zoomed')
drawBezier()
my_window.mainloop()
