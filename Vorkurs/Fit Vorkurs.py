
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.odr as odr


x = np.array([5, 7, 10, 12])
xError = np.array([0.1, 0.1, 0.1, 0.1])
y = np.array([0.318, 0.473, 0.703, 0.870])
yError = np.array([0.005, 0.011, 0.024, 0.038])


# Plotparameter
xAnfang = 4
xEnde = 14
yAnfang = 0.3
yEnde = 0.9


# Curve Fit
def f(Parameter, x):
    return Parameter[0]*x + Parameter[1]
model = odr.Model(f)
mydata = odr.RealData(x, y, sx=xError, sy=yError)
data = odr.ODR(mydata, model, beta0=[1., 2.])
out = data.run()
out.pprint()

xCalc = np.linspace(xAnfang, xEnde, len(x))
yCalc = f(out.beta, xCalc)


# R^2 Wert Berechnung
residuals = y- f(out.beta, x)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
rsquared = 1 - (ss_res / ss_tot)









fig, ax = plt.subplots()
plt.plot(xCalc, yCalc, '-')
plt.errorbar(x, y, xerr=xError, yerr=yError, fmt='.', linewidth=3, capsize=3, c='black') 
plt.plot(xCalc, yCalc, c='red')
plt.grid()
plt.xlabel("Motorspannung in V")
plt.ylabel("Frequenz in Hz")
plt.title(f'Frequenz gegen Spannung mit $R^2 =$ {rsquared:.4f}')
plt.xlim(xAnfang, xEnde)
plt.ylim(yAnfang, yEnde)
plt.show()
