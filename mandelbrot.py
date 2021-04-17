import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from PIL import Image
import threading

plt.style.use('dark_background')

T = 50
#n = 2160 #1000
#m = 3840 #5 * n // 4
n = 4000
m = 4000
p = 2

viridis = cm.get_cmap('viridis', T)
cmap = cm.get_cmap('viridis', T)

class myThread(threading.Thread):
    def __init__(self, C):
        threading.Thread.__init__(self)
        self.C = C
        self.Z = np.zeros_like(C)
        self.M = np.zeros(C.shape)
        #self.P = np.sqrt((X-.25)**2 + Y**2)
        #self.mask = np.logical_and(X <= self.P - 2*self.P**2 + .25, (X-1)**2 + Y**2 <= 1 / 16)

    def run(self):
        for i in range(T):
            #index = np.logical_and(self.M == 0, np.logical_not(self.mask))
            index = self.M == 0
            print(i, np.sum(index))
            self.Z[index] = self.Z[index] ** 2 + self.C[index]
            self.M[index] = np.where(np.logical_and(index[index], np.abs(self.Z[index]) > 2), i + 1, self.M[index])


def mandelbrot(x0=-3., x1=1.3, y0=-1.2, y1=1.2, T=100, n=2*2160, m=2*3840, p=2):

    cmap = cm.get_cmap('bone', 2048)
    x = np.linspace(x0, x1, m)
    y = np.linspace(0, y1, n // 2)
    X, Y = np.meshgrid(x, y)
    C = X + 1.j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape)
    P = np.sqrt((X - .25) ** 2 + Y ** 2)
    cardioidMask = np.logical_and(X <= P - 2 * P ** 2 + .25, (X - 1) ** 2 + Y ** 2 <= 1 / 16)
    for i in range(T):
        print(i)
        index = np.logical_and(M == 0, np.logical_not(cardioidMask))
        M[index] = np.where(np.logical_and(index[index], np.abs(Z[index]) > 2), i+1, M[index])
        if i == T-1: break
        Z[index] = Z[index]**p + C[index]

    index = M != 0
    logZ = np.log(np.abs(Z[index]))
    nu = np.log(logZ / np.log(2)) / np.log(2)
    M[index] = M[index] - nu

    img = np.zeros((n - 1, m, 4))
    img[n // 2 - 1:] = cmap(np.sqrt(M / np.max(M)))
    img[n // 2 - 1:, :, :-1] = np.where(M[:, :, None] == 0, 0, img[n // 2 - 1:, :, :-1])
    img[:n // 2 - 1] = img[:n // 2 - 1:-1]
    plt.imsave('mandelbrottest.png', img)
    #plt.imshow(img)
    #plt.imshow(ma.masked_equal(img, 0), cmap='bone', origin='lower')
    #plt.xticks(np.linspace(0, n, 10), np.round(np.linspace(x0, x1, 10), 2))
    #plt.yticks(np.linspace(0, n, 10), np.round(np.linspace(y0, y1, 10), 2))
    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()


def mandelbrot2(x0=-3., x1=1.3, y0=-1.2, y1=1.2, T=100, n=2160, m=3840, p=2):
    cmap = cm.get_cmap('bone', 1024)
    x = np.linspace(x0, x1, m)
    y = np.linspace(0, y1, n // 2)
    X, Y = np.meshgrid(x, y)
    C = X + 1.j * Y
    Z = np.zeros_like(C)
    escapeTime = np.zeros(C.shape, dtype=int)
    P = np.sqrt((X - .25) ** 2 + Y ** 2)
    cardioidMask = np.logical_and(X <= P - 2 * P ** 2 + .25, (X - 1) ** 2 + Y ** 2 <= 1 / 16)
    for i in range(T):
        print(i)
        index = np.logical_and(escapeTime == 0, np.logical_not(cardioidMask))
        escapeMask = np.logical_and(index[index], np.abs(Z[index]) > 2)
        #if np.sum(escapeMask) == 0:
        #    print(i+1)
        #    break
        escapeTime[index] = np.where(escapeMask, i + 1, escapeTime[index])
        Z[index] = Z[index] ** p + C[index]

    #index = np.logical_not(index)
    index = escapeTime != 0

    NumIterations = np.histogram(escapeTime, T+1, (0,T+1))[0]

    total = np.sum(NumIterations)

    hue = np.zeros(escapeTime.shape)
    for xi in range(m):
        for yi in range(n//2):
            for t in range(0, escapeTime[yi,xi]+1):
                hue[yi,xi] += NumIterations[t] / total

    logZ = np.log(np.abs(Z[index]))
    nu = np.log(logZ / np.log(2)) / np.log(2)
    hue[index] = hue[index]*T - nu

    img = np.zeros((n - 1, m, 3))
    img[n // 2 - 1:] = cmap(hue/np.max(hue))[:,:,:-1]
    img[n // 2 - 1:] = np.where(hue[:,:,None] == 0, 0, img[n // 2 - 1:])
    img[:n // 2 - 1] = img[:n // 2 - 1:-1]
    plt.imsave('mandelbrottest.png', img)


def mandelbrot_distest():
    a = 0.0005
    x0 = -0.7463 - a
    x1 = -0.7463 + a
    y0 =  0.1102 - a
    y1 =  0.1102 + a
    #R = 0.32
    #x0 = -0.925 - R
    #x1 = -0.925 + R
    #y0 = 0.266 - R
    #y1 = 0.266 + R
    #x0 = -3.
    #x1 = 1.3
    #y0 = -1.2
    #y1 = 1.2


    N = 101
    #n = 2160
    #m = 3840
    n = 4000
    m = 4000
    h = (x1 - x0) / m
    print(h)

    x = np.linspace(x0, x1, m)
    #y = np.linspace(0., y1, n // 2)
    y = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(x, y)
    C = X + 1.j * Y
    Pc = np.copy(C)
    dPc = np.ones_like(C)
    #escape = np.zeros(C.shape, dtype=bool)
    #P = np.sqrt((X - .25) ** 2 + Y ** 2)
    #cardioidMask = np.logical_and(X <= P - 2 * P ** 2 + .25, (X - 1) ** 2 + Y ** 2 <= 1 / 16)

    mask = np.ones(C.shape, dtype=bool)
    for i in range(N):
        print(i)
        mask[mask] = np.abs(Pc[mask]) < 2
        dPc[mask] = 2*Pc[mask]*dPc[mask] + 1
        Pc[mask] = Pc[mask]**2 + C[mask]

    absPc = np.abs(Pc[mask])
    b = 2*absPc*np.log(absPc)/np.abs(dPc[mask])

    M = np.zeros(C.shape, dtype=int)
    M[mask] = np.where(b < h/4, 1, 0)

    cmap = cm.get_cmap('binary', 2)
    img = cmap(M)[:,:,:-1]
    #img = np.zeros((n - 1, m, 3))
    #img[n // 2 - 1:] = cmap(M)[:,:,:-1]
    #img[:n // 2 - 1] = img[:n // 2 - 1:-1]
    plt.imsave('mandelbrottest.png', img)


def julia(x0=-3.2, x1=3.2, y0=-1.8, y1=1.8, T=300, R=2, n=1080, m=1920):

    frame = 0
    #sample = np.tan(np.linspace(-1,1, 48))
    sample = 1/ np.linspace(1,2, 121)
    sample -= sample[0]
    sample *= np.pi / sample[-1]
    #plt.hist(sample)
    #plt.show()
    #return
    for a in sample:
        frame += 1
        c =  0.7885*np.exp(1j*a)
        x = np.linspace(x0, x1, m)
        y = np.linspace(y0, y1, n)
        X, Y = np.meshgrid(x, y)
        Z = X + 1.j * Y
        J = np.zeros(Z.shape)
        maxi = 0

        cmap = cm.get_cmap('cubehelix', 1024)
        for i in range(T):
            #print(i)
            index = J == 0
            Z[index] = Z[index]**2 + c
            mask = np.logical_and(index[index], np.abs(Z[index]) > R**2)
            if np.sum(mask) == 0:
                print(i+1)
                break
            J[index] = np.where(mask, i+1, J[index])
            maxi = max(maxi, i+1)

        index = np.logical_not(index)
        #J[index] = np.clip(J[index], 0, 1024)
        logZ = np.log(np.abs(Z[index]))
        nu = np.log(logZ / np.log(2)) / np.log(2)
        J[index] = J[index] - nu

        img = cmap(J/np.max(J))[:,:,:-1]
        #img = cmap(np.clip((J-1) / 200, 0, 1))[:,:,:-1]
        img = np.where(J[:, :, None] == 0, 0, img)
        plt.imsave('julanim/julanim%s.png' % frame, img)


#x0, x1 = -1.8, .6
#y0, y1 =  -1.2, 1.2

#x0, x1 = -1.05, -0.55
#y0, y1 =  0, .5

#x0, x1 = -3, 1.3
#y0, y1 = -1.2, 1.2

#x0, x1 = -2.1, 1.1
#y0, y1 = -1.6, 1.6

#x0, x1 = -2.1, .9
#y0, y1 =  -1.2, 1.2

#x = np.linspace(-1.8, .6, n)
#y = np.linspace(-1.2, 1.2, n)

#x = np.linspace(x0, x1, 3840)
#y = np.linspace(y0, y1, 2160)

#x = np.linspace(x0, x1, m)
#y = np.linspace(0, y1, n//2)

#X, Y = np.meshgrid(x, y)
#C = X + 1.j * Y
#Z = np.zeros_like(C)
#M = np.zeros(C.shape)
#P = np.sqrt((X-.25)**2 + Y**2)
#mask = np.logical_and(X <= P - 2*P**2 + .25, (X-1)**2 + Y**2 <= 1 / 16)

#t0 = time()
#for i in range(T):
#    print(i)
#    index = np.logical_and(M == 0, np.logical_not(mask))
#    Z[index] = Z[index]**p + C[index]
#    M[index] = np.where(np.logical_and(index[index], np.abs(Z[index]) > 2), i+1, M[index])

#t = time() - t0

#print(t)



#img = np.zeros((n-1, m, 4))
#img[n//2-1:] =  cmap(M / np.max(M))
#img[n//2-1:,:,:-1] = np.where(M[:,:,None] == 0, 0, img[n//2-1:,:,:-1])
#img[:n//2-1] = img[:n//2-1:-1]
#plt.imsave('mandelbrottest.png', img)


for i in range(1, 120):
    print(121-i, '->', 121+i)
    im = Image.open('julanim/julanim%s.png' % (121 - i))
    im = im.transpose(method=Image.FLIP_LEFT_RIGHT)
    im.save('julanim/julanim%s.png' % (i + 121))

#julia()