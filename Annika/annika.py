import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np

Punkte = 10000

Radius = np.random.rand(Punkte)
phi = 2 * np.pi * np.random.rand(Punkte)
theta = np.pi * np.random.rand(Punkte)

x = Radius * np.sin(theta) * np.cos(phi)
y = Radius * np.sin(theta) * np.sin(phi)

xWerte = []
yWerte = []

FxWerte = []
FyWerte = []

Maß = 0.5
i = 0

ax, plt=plt.subplots()


xWerte = []
yWerte = []
FxWerte = []
FyWerte = []
RadiusWerte = []

for i in range(Punkte):
    if (x[i] < Maß and y[i] < Maß and x[i] > - Maß and y[i] > - Maß):
        xWerte.append(x[i])
        yWerte.append(y[i])
        RadiusWerte.append(Radius[i])
    else:
        FxWerte.append(x[i])
        FyWerte.append(y[i])

    
    
plt.scatter(FxWerte, FyWerte, s=5, c='b', marker='.')
plt.scatter(xWerte, yWerte, s=5, c='r', marker='.')
plt.set_xlim(-1,1)
plt.set_ylim(-1, 1)
plt.set_aspect('equal', adjustable='box')
ax.savefig('Annika\\Kugel.pdf')
plt.cla()
plt.plot(RadiusWerte, xWerte)
plt.plot(RadiusWerte, yWerte)

ax.savefig('Annika\\Kugelplot.pdf')