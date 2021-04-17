import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np


G = 2.95912208286e-4
M = np.array([0.000954786104043,
              0.000285583733151,
              0.0000437273164546,
              0.0000517759138449,
              1/1.3*1e-8,
              1.00000597682])

def g(x,y):
    z = x - y
    return -G * z/(np.sum(z**2)**(3/2))

def F(X):
    Y = np.zeros_like(X)
    for i in range(5):
        Y[i] = X[i+6]
        for j in range(6):
            if j != i:
                Y[i+6] += M[j] * g(X[i],X[j])
    return Y

def EulerExp(X, dt):
    return X + F(X)*dt

def RK4(X, dt):
    k1 = F(X)*dt
    k2 = F(X + k1/2)*dt
    k3 = F(X + k2/2)*dt
    k4 = F(X + k3)*dt
    return X + (k1 + 2*k2 + 2*k3 + k4)/6

def simulateSolarSystem(X0, T, dt, integ_func=RK4):
    n = T // dt
    X = np.zeros((n,12,3))
    X[0] = X0
    for i in range(1,n):
        X[i] = integ_func(X[i-1],dt)

    return X

def animateSolarSystem(X):
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.axis('off')
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_zlim(-30,30)

    C = ['darkorange','khaki','steelblue','darkturquoise','saddlebrown']
    sun = [ax.plot(X[:1,5,0],X[:1,5,1],X[:1,5,2],'yo')[0]]
    planets = [ax.plot(X[:1,i,0],X[:1,i,1],X[:1,i,2],'o',color=C[i])[0] for i in range(5)]
    orbits  = [ax.plot(X[:1,i,0],X[:1,i,1],X[:1,i,2], color=C[i],lw=.8)[0] for i in range(5)]

    def animate(i, p):
        for k in range(5):
            p[k].set_data(X[i,k,0],X[i,k,1])
            p[k].set_3d_properties(X[i,k,2])

            p[k+5].set_data(X[:i,k,0],X[:i,k,1])
            p[k+5].set_3d_properties(X[:i,k,2])

        p[-1].set_data(X[i, 5, 0], X[i, 5, 1])
        p[-1].set_3d_properties(X[i, 5, 2])
        return p
    ssAnimation = animation.FuncAnimation(fig, animate, frames=len(X),
                                          fargs=(planets+orbits+sun,),interval=20)

    plt.show()


X0 = np.reshape(np.array([[-3.5023653],[-3.8169847],[-1.5507963],
               [9.0755314],[-3.0458353],[-1.6483708],
               [8.3101420],[-16.2901086],[-7.2521278],
               [11.4707666],[-25.7294829],[-10.816945],
               [-15.5387357],[-25.2225594],[-3.1902382],
               [0],[0],[0],
               [0.00565429],[-0.00412490],[-0.00190589],
               [0.00168318],[0.00483525],[0.00192462],
               [0.00354178],[0.00055029],[0.00055029],
               [0.00288930],[0.00039677],[0.00039677],
               [0.00276725],[-0.00136504],[-0.00136504],
               [0],[0],[0]]),(12,3))

X1 = np.reshape(np.array([[-3.5023653],[-3.8169847],[-1.5507963],
               [2.0755314],[-10.0458353],[-0.6483708],
               [8.3101420],[-16.2901086],[-7.2521278],
               [11.4707666],[-25.7294829],[-10.816945],
               [-15.5387357],[-25.2225594],[-3.1902382],
               [0],[0],[0],
               [0.00565429],[-0.00412490],[-0.00590589],
               [0.00168318],[0.00483525],[0.00192462],
               [0.00354178],[0.00055029],[0.00055029],
               [0.00088930],[0.00039677],[0.00339677],
               [0.00276725],[-0.00136504],[-0.00136504],
               [0],[0],[0]]),(12,3))

X = simulateSolarSystem(X0, 100000, 100)

animateSolarSystem(X)
