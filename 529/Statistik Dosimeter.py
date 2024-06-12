import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np

Punkte = 10000

Radius = 4.5*np.sqrt(np.random.rand(Punkte))

Winkel = 2 * np.pi * np.random.rand(Punkte)

x = Radius * np.cos(Winkel)
y = Radius * np.sin(Winkel)

xWerte = []
yWerte = []

FxWerte = []
FyWerte = []

Ungenauigkeit = 2.2
while len(xWerte)/Punkte < 0.683:
    print(len(xWerte)/Punkte)
    xWerte = []
    yWerte = []
    FxWerte = []
    FyWerte = []
    for i in range(Punkte):
        if x[i] < Ungenauigkeit:
            if x[i] > -Ungenauigkeit:
                xWerte.append(x[i])
                yWerte.append(y[i])
            else:
                FxWerte.append(x[i])
                FyWerte.append(y[i])
        else:
                FxWerte.append(x[i])
                FyWerte.append(y[i])
    Ungenauigkeit = Ungenauigkeit + 0.001
    
    
plt.scatter(FxWerte, FyWerte, s=5, c='b', marker='.')
plt.scatter(xWerte, yWerte, s=5, c='r', marker='.')
print(f'Die Messungenauigkeit des Radius beträgt für 1 Sigma etwa {Ungenauigkeit:1f} mm')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('P5\\529\\Statistik Dosimeter.png')
plt.show()
