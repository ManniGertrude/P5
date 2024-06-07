import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np

Punkte = 100000

Radius = 4.5*np.sqrt(np.random.rand(Punkte))

Winkel = 2 * np.pi * np.random.rand(Punkte)

x = Radius * np.cos(Winkel)
y = Radius * np.sin(Winkel)

xWerte = []
yWerte = []
Ungenauigkeit = 1
while len(xWerte)/Punkte < 0.683:
    xWerte = []
    yWerte = []
    for i in range(Punkte):
        if x[i] < Ungenauigkeit:
            if x[i] > -Ungenauigkeit:
                xWerte.append(x[i])
                yWerte.append(y[i])
    Ungenauigkeit = Ungenauigkeit + 0.05
    
    
    
plt.scatter(xWerte, yWerte, s=5)
print(f'Die Messungenauigkeit des Radius beträgt für 1 Sigma etwa {Ungenauigkeit:1f} mm')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
