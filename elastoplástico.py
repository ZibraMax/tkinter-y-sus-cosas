import math
from tkinter import Tk, Canvas, W, E, NW
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from scipy.interpolate import interp1d
import time
import numpy as np


def getCubic(L, disp):
    K = np.array([[3*L**2, 2*L], [L**3, L**2]])
    F = np.array([[np.arctan2(disp, L)], [disp]])
    return np.linalg.solve(K, F)[:, 0]


def graficaParabola(u, dash, widthline, fill='', fillc=''):
    global width, height, mult, L, m, k, multu
    centrox, centroy = width/2, height-100
    r = m*2
    up = u*mult*multu
    Lp = L*mult
    theta = np.arctan2(up, Lp)
    Lp = Lp*np.cos(theta)
    up = up*np.cos(theta)
    a, b = getCubic(Lp, up)
    x, y = 0, 0
    for i in range(20+1):
        equis = (Lp)/20*i
        xi, yi = equis, a*equis**3+b*equis**2
        my_canvas.create_line(y+centrox, centroy-x, yi+centrox,
                              centroy-xi, dash=dash, fill=fill, width=widthline)
        x, y = xi, yi
    my_canvas.create_oval(y-r+centrox, centroy-(x-r),
                          y + r+centrox, centroy-(x+r), fill=fillc)


def graficarBola():
    my_canvas.delete("all")
    graphAcel()
    ponerTextos()
    global width, height, mult, u, L, m, k, multu, uy, up, ue
    graficaParabola(0, dash=[3, 3], widthline=1,
                    fill='black', fillc='lightgray')
    graficaParabola(-uy, dash=[3, 3], widthline=1, fill='black', fillc='white')
    graficaParabola(uy, dash=[3, 3], widthline=1, fill='black', fillc='white')
    graficaParabola(up, dash=[3, 3], widthline=1,
                    fill='pink', fillc='orange')
    graficaParabola(ue, dash=[3, 3], widthline=1,
                    fill='pink', fillc='green')
    graficaParabola(u, dash=[], widthline=max(
        1, int(k/100)), fill='red', fillc='blue')


def editPoint(event):
    global editando
    editando = not editando
    if not editando:
        drawBezier()


def drawBezier():
    global editando, u, v, acel, L, z, f, m, k, dt, t, height, width, T, U, V, ACEL, tol, betai, betas, uy, ksh, qsh, up, ue, FUERZA
    acel = -f(0)*9.81
    beta = eval(betas[betai])
    qy = k*uy
    kp = k
    ue = u
    while not editando:
        omega = np.sqrt(k/m)
        t += dt
        err = 1
        ud1 = 0
        u1 = 0
        udd1 = acel
        while err > tol:
            ud1 = v + ((acel + udd1)/2)*dt
            ah = (1-2*beta)*acel+2*beta*udd1
            u1 = ue + v * dt + 1/2*ah*dt**2
            udd2 = (-m*f(t)*9.81 - 2*m*omega*z*ud1-k*u1)/m
            err = abs(udd1-udd2)
            udd1 = udd2
        v = ud1
        ue = u1
        acel = udd1

        q = k*(ue-up)
        flujo = abs(q-qsh)-qy
        if flujo < 0:
            kp = k
        else:
            deltaup = flujo/(k+ksh)*np.sign(q-qsh)
            deltaqsh = ksh*deltaup
            kp = ksh
            q = q-k*deltaup
            up = up + deltaup
            qsh = qsh + deltaqsh
        u = ue+up
        U += [ue+up]
        V += [v]
        T += [t]
        ACEL += [acel]
        FUERZA += [q]
        # time.sleep(dt/10)
        graficarBola()
        graficasNewmark()
        my_canvas.update()


def movePoint(event):
    global editando, u, L, width, height, U
    if editando:
        centrox, centroy = width/2, height-100
        my_canvas.delete("all")
        x = centrox-event.x
        y = centroy-event.y
        P[0] = x
        P[1] = y
        u, L = -x/mult/multu, y/mult
        U[-1] = u
        graficarBola()
        graficasNewmark()


def ponerTextos():
    global z, k, m, dt, height, multu, strt, betai, betas, uy, width
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
    my_canvas.create_text(100, height-210, fill="black",
                          font='20', text=f"beta={betas[betai]}", anchor=W)
    my_canvas.create_text(100, height-230, fill="black",
                          font='20', text=f"uy={format(uy,'.4f')}", anchor=W)
    my_canvas.create_text(200, height-230, fill="black",
                          font='20', text=f"fy={format(uy*k,'.2f')}", anchor=W)
    my_canvas.create_text(100, height-250, fill="orange",
                          font='20', text=f"Plásticas", anchor=W)
    my_canvas.create_text(100, height-270, fill="green",
                          font='20', text=f"Elásticas", anchor=W)
    my_canvas.create_text(100, 50, fill="black",
                          font='20', text=strt, anchor=NW)


def graficasNewmark():
    global U, V, ACEL, T, m, f, FUERZA, graficasCompletas
    x0 = 100
    y0 = height-90
    b = 300
    h = 100
    createGraph(width-b-x0, y0-150, b, h, T, U,
                title='u [m]', alert=graficasCompletas)
    createGraph(width-b-x0, y0-2*150, b, h, T, V,
                title='v [m/s]', alert=graficasCompletas)
    createGraph(width-b-x0, y0-3*150, b, h, T,
                ACEL, title='a [m²/s]', alert=graficasCompletas, time=False)
    createGraph(width-b-x0, y0-4*150, b, h, T,
                FUERZA, title='F [kN]', time=False, alert=graficasCompletas)
    createGraph(width-b/2-x0, y0-5*150, b/2, h, U,
                FUERZA, xaxis=True, time=True, title='Histeresis', alert=graficasCompletas)
    # print(max(np.array(ACEL)/(9.81)))


def createGraph(x0, y0, b, h, X, Y, maxs=None, color='red', title='', alert=False, time=False, xaxis=False):
    XC = []
    YC = []
    b = 2*b
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
    xf = x0+b/2
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
    xmax = max(abs(xmax), abs(xmin))
    if ymax == 0:
        ymax = 1
    if xmax == 0:
        xmax = 1

    def z(x): return (x)/ymax
    def zx(x): return (x)/xmax
    X = [zx(i) for i in X]
    Y = [z(i) for i in Y]
    my_canvas.create_line(x0, y0+10*xaxis, x0, yf-10 *
                          xaxis, fill='gray', width=1)
    my_canvas.create_line(x0-b/2*xaxis, ym, xf, ym, fill='gray', width=1)
    my_canvas.create_text(x0-b/2*xaxis-5, ym, fill="black",
                          font='20', text=title, anchor=E)
    if time:
        my_canvas.create_text(x0-b/2*xaxis-20, yf-20, fill="black",
                              font='20', text=f"t={format(t,'.2f')}", anchor=W)

    my_canvas.create_text(x0-b/2*xaxis-5, y0, fill="black",
                          font='5', text=format(-ymax, '.4f'), anchor=E)
    my_canvas.create_text(x0-b/2*xaxis-5, y0-h, fill="black",
                          font='5', text=format(ymax, '.4f'), anchor=E)
    if xaxis:
        my_canvas.create_text(x0+b/2*xaxis+5, y0, fill="black",
                              font='5', text='+/-'+format(xmax, '.4f'), anchor=E)
    for i in range(len(X)-1):
        my_canvas.create_line(x0+X[i]*b/2, ym-Y[i]*h/2, x0 +
                              X[i+1]*b/2, ym-Y[i+1]*h/2, fill=color, width=2)


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
    createGraph(width-b-x0, y0, b, h, X, Y, maxs, color='blue', title='ag [g]')


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
    global editando, actual, f, u, v, t, U, T, V, ACEL, acel, m, ue, up, FUERZA, graficasCompletas
    if e.char.lower() == 'a':
        U, T, V, ACEL, FUERZA = [], [], [], [], []
        u, ue, up, v, t, acel = 0, 0, 0, 0, 0, 0
        def f(x): return 0
        importarArchivo()
    if e.char.lower() == 'r':
        u, ue, up, v, acel, t = 0, 0, 0, 0, 0, 0
        def f(x): return 0
        U, T, V, ACEL, FUERZA = [], [], [], [], []
    if e.char.lower() == 't':
        u, ue, up, v, acel, t = 0, 0, 0, 0, 0, 0
        U, T, V, ACEL, FUERZA = [], [], [], [], []
    elif e.char.lower() == 'g':
        graficasCompletas = not graficasCompletas
    else:
        actual = e.char.lower()


def wheel(event):
    global z, k, m, dt, height, actual, multu, editando, betai, uy
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
    elif actual == 'b':
        betai += np.sign(delta)
        betai = max(betai, 0)
        betai = min(betai, 2)
    elif actual == 'y':
        uy += 0.005*np.sign(delta)
        uy = max(uy, 0)
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
z = 0.05
L = 1.5
z = 0.05
m = 20
k = 1500
kh = 0*k
ksh = kh*k/(k-kh)
qsh = 0
dt = 0.05
up = 0
ue = 0
multu = 100
data = None
tol = 1*10**(-8)
betas = ['1/8', '1/6', '1/4']
betai = 1
uy = 0.005
graficasCompletas = True
U = []
V = []
T = []
ACEL = []
FUERZA = []
P = [u*mult/multu, L*mult]
strt = "Controles:\nClick: Mover la masa\nA: Seleccionar archivo de aceleración\n\nPara cambiar las propiedades, use una de las siguientes letras\ny cambielas usando la rueda del mouse:\n\nK: Rigidez\nM: Masa\nZ: Amortiguamiento\nd: Paso en el tiempo\nu: Multiplicador de desplazamientos (solo para graficar)\nb: Beta de Newmark\ny: uy\n\nR: Reiniciar todo\nT: Reiniciar tiempo\ng: Modo de gráficas rápidas"
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
# importarArchivo()
drawBezier()
my_window.mainloop()
