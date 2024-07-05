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
# Path = 'C:\\Users\\kontr\\Desktop\\Github\\P5\\521\\'
data = pd.read_csv(f'{Path}spectrum.txt', sep='\t',header=0, names=['x', 'y'])
xData = data['x'].values
yData = data['y'].values
yWerte = np.array(np.zeros(len(yData)))


def gaus(Para, x): 
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2) + Para[3]*x + Para[4]


def gausfit(func, x, y, farbe, beta, Name):
    model = odr.Model(func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=1000)
    out = myodr.run()
    fy = func(out.beta, x)
    plt.plot(x, fy, c=farbe, label=Name)
    print(f'$Parameter {i+1}:', out.beta, out.sd_beta,  '$')
    print('$R^2 =', r2_score(y, fy), '$')
    print()
    return out.beta

ax, plt = plt.subplots()


# Gaus-Fits

Farbe = ['firebrick', 'sienna', 'darkgoldenrod', 'darkolivegreen', 'steelblue', 'orchid']
Beta = [[44000, 300, 33, 1, 1], [10000, 600, 100, 1, 1], [80000, 900, 130, 1, 1], [10000, 2200, 1500, 1, 1], [40000, 3500, 10, 1, 1], [70000, 4000, 15, 1, 1]]
Fenster = [170, 480, 510, 690, 790, 1120, 1100, 2800, 2800, 3600, 3600, 4500]

for i in range(len(Farbe)):
    TempxData = np.where((xData > Fenster[2*i+1]) | (xData < Fenster[2*i]), 0, xData)
    TempyData = np.where((xData > Fenster[2*i+1]) | (xData < Fenster[2*i]), 0, yData)
    if i == 3:
        yWerte = yWerte 
    else:
        out = gausfit(gaus, xData[Fenster[2*i]:Fenster[2*i+1]], yData[Fenster[2*i]:Fenster[2*i+1]], Farbe[i], Beta[i], f'Gauskurve {i+1}')
        yWerte = yWerte + gaus(out, xData)


plt.scatter(xData, yData, s=2, c='navy', label='Messwerte')
# plt.plot(xData, yWerte, c='lavender', label='Gauskurven' )
plt.legend()
plt.set_xticks(np.linspace(0, 8000, 17))
plt.set_xticks(np.linspace(0, 8000, 81), minor=True, alpha=0.3)
plt.set_xlim(-50, 4500)
plt.set_ylim(-10, max(yData)*1.05)
plt.axis()
plt.grid()
ax.savefig(f'{Path} Testspektrum 3')

