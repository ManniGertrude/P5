import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit 
import os
import scipy.odr as odr
import math
from sklearn.metrics import r2_score


Path = 'C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\521\\'
data = pd.read_csv(f'{Path}spectrum.txt', sep='\t',header=0, names=['x', 'y'])


def gaus(Para, x): 
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2)


def gausfit(func, x, y, farbe, beta, Name):
    model = odr.Model(func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=1000)
    out = myodr.run()
    fy = func(out.beta, x)
    plt.plot(x, fy, c=farbe, label=Name )

    print('$Parameter:', out.beta, out.sd_beta,  '$')
    # print('$R^2 =', r2_score(y, fy), '$')
    return out.beta


Farbe = ['firebrick', 'sienna', 'darkgoldenrod', 'darkolivegreen', 'steelblue', 'orchid']
Beta = [[44000, 300, 33], [10000, 600, 100], [80000, 900, 130], [10000, 2200, 1500], [40000, 3500, 10], [70000, 4000, 15]]
Fenster = [180, 450, 450, 700, 700, 1100, 1100, 3000, 2800, 3500, 3500, 5000]


ax, plt = plt.subplots()
yWerte = np.array(np.zeros(len(data['y'])))
plt.scatter(data['x'], data['y'], s=2, c='navy', label='Messwerte')
for i in range(len(Farbe)):
    out = gausfit(gaus, data['x'][Fenster[2*i]:Fenster[2*i+1]], data['y'][Fenster[2*i]:Fenster[2*i+1]], Farbe[i], Beta[i], f'Gauskurve {i+1}')
    yWerte = yWerte + gaus(out, data['x'])


plt.plot(data['x'], yWerte, c='black', label='Gauskurven' )
plt.legend()
plt.grid()
ax.savefig(f'{Path} Testspektrum')