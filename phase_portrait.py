import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('dark_background')

def f(Z, t):
    x, y = Z
    return 2*x*(1-y), 2*y-x**2+y**2

def phase_portrait_test():
    a, b = 5, 7
    x = np.linspace(-a, a, 50)
    y = np.linspace(-b, b, 50)

    X, Y = np.meshgrid(x, y)
    t = 0

    u, v = np.zeros(X.shape), np.zeros(Y.shape)

    NI, NJ = Y.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]
            u[i, j], v[i, j] = f((x, y), t)

    U = u / np.sqrt(u**2+v**2)
    V = v / np.sqrt(u**2+v**2)

    x0 = np.linspace(-a, a, 100)
    y1 = -np.sqrt(x0**2+1)-1
    y2 =  np.sqrt(x0**2+1)-1

    y3 = np.ones_like(x0)
    y4 = np.zeros_like(x0)

    y5 =  x0/np.sqrt(3)
    y6 = -x0/np.sqrt(3)

    #plt.plot(x0,y1, x0, y2, color='orange', lw=1)
    plt.plot(x0,y3, y4, x0, color='b', lw=.5)
    plt.plot(x0,y5, x0, y6, color='b', lw=.5)
    plt.plot((-np.sqrt(3),np.sqrt(3),0,0),(1,1,0,-2), 'rx')
    #plt.plot((-np.sqrt(3),np.sqrt(3),0,0),(1,1,0,-2), 'rx')


    #Q = plt.streamplot(X, Y, u, v, color='gray', density=2.5, linewidth=.75, arrowsize=.5)
    Q = plt.quiver(X, Y, U, V, np.sqrt(u**2+v**2), color='r', linewidth=.75, cmap='plasma', scale=50)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-a-1, a+1])
    plt.ylim([-b-1, b+1])

    print(plt.rcParams['axes.prop_cycle'])

    plt.show()

def phase_portrait(f,
                   a, b,
                   a0=0, b0=0,
                   delta=.3,
                   x0y0=None,
                   t0=0,
                   xlabel='x',
                   ylabel='y',
                   normalise_arrows=True,
                   animate=False,
                   stream=False,
                   trace_orbit=True,
                   cmap='plasma'):
    """
    Plots the phase portrait of the 2D vector field characterised by
    the differential equation Z'(t) = f(Z(t), t)

    f : function that takes as arguments a tuple of floats: Z=(x,y) and a float: t

    a : size of portrait along the horizontal axis

    b : size of portrait along the vertical axis

    a0, b0 : coordinates of center of portrait

    points : number of points calculated in the phase portrait

    t0 : initial time of animation

    animate : if true animates the phase portrait as time evolves

    normalise_arrows : if true draws all arrows with equal length

    stream : if True makes a stream plot, else makes a quiver plot

    """


    fig = plt.figure(figsize=plt.figaspect(b/a))
    ax = fig.add_subplot(111)
    ax.set_aspect(2*b/a)

    #x = np.linspace(x0-a/2, x0+a/2, points)
    #y = np.linspace(y0-b/2, y0+b/2, points)


    x = np.arange(a0-a/2, a0+a/2, delta)
    y = np.arange(b0-b/2, b0+b/2, delta)

    X, Y = np.meshgrid(x, y)

    if t0 is None:
        t = 0
    else:
        t = t0

    U, V = f((X,Y), t)
    norm = np.sqrt(U ** 2 + V ** 2)
    if normalise_arrows:
        U = U / norm
        V = V / norm

    if stream:
        Q = ax.streamplot(X, Y, U, V, color=norm, cmap=cmap,
                           density=2.5, linewidth=.75, arrowsize=.5)
    else:
        Q = ax.quiver(X, Y, U, V, norm, color='r', linewidth=.75, cmap=cmap, scale=50)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(a0-a*.55, a0+a*.55)
    ax.set_ylim(b0-b*.55, b0+b*.55)

    if x0y0 is not None:
        global x0, y0, x_orbit, y_orbit, dt
        x0, y0 = x0y0
        dt = .1
        x_orbit = [x0]
        y_orbit = [y0]
        point, = ax.plot([], [], 'r.')
        orbit, = ax.plot([], [], 'r', lw=.6)


        def init():
            point.set_data([], [])
            orbit.set_data([], [])
            return point, orbit

        def animate(i):
            global x0, y0, dt
            if abs(x0) > a0+.5*a or abs(y0) > b0+.5*b:
                return point,
            dx0, dy0 = f((x0,y0), t)
            x0 += dx0 * dt
            y0 += dy0 * dt
            point.set_data(x0, y0)
            if trace_orbit:
                x_orbit.append(x0)
                y_orbit.append(y0)
                orbit.set_data(x_orbit,y_orbit)
            return point, orbit

        from time import time

        t0 = time()
        animate(0)
        t1 = time()
        interval = 500 * dt - (t1 - t0)

        ani = animation.FuncAnimation(fig, animate, frames=300,
                                      interval=interval, blit=True, init_func=init)


    plt.show()

    return ax, fig

def g(Z, t):
    omega = 1
    mu = .25
    x, y = Z
    return y, -omega**2*np.sin(x) - mu*y

phase_portrait(g, 20, 10, x0y0=(-9,5), xlabel=r"$\theta$", ylabel=r"$\theta'$", stream=True, cmap='viridis')

