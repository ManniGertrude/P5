import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit 
import os
import scipy.odr as odr
import math
from sklearn.metrics import r2_score



x = np.array([0, 16, 32, 48, 64])
y = np.array([247.9, 518.5, 798.3, 1078, 1271.2])
xErr = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
yErr = np.array([0.6, 0.5, 0.5, 0.5, 0.5])

def func(Para, x):
    return Para[0] * x + Para[1]
 
model = odr.Model(func)
mydata = odr.RealData(x, y)# , sx = xErr, sy = yErr
myodr = odr.ODR(mydata, model, beta0=[1., 1.], maxit=1000)
out = myodr.run()

fy = func(out.beta, x)
rsquared = r2_score(y, fy)
# out.pprint()
print('$Parameter:', out.beta, out.sd_beta,  '$')
print('$R_{lin}^2 =',rsquared, '$')





ax, plt = plt.subplots()
plt.grid()
plt.errorbar(x, y,marker='+', color='navy', markersize=8, linestyle='none', label='Messwerte' )
plt.plot(x, fy, c='slateblue',label = 'Ausgleichsgrade')
plt.set(xlabel='Verzögerung in ns', ylabel='Erwartungswert der Promptkurve in Kanälen')
plt.legend()
ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Zeitauflösung.pdf')
ax.show
