import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np



Strom = np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Rate = np.array([1174, 1749, 1979, 2161, 2309, 2449, 2578, 3110, 3495, 3828, 4115, 4647, 5192, 5607, 6109, 6470, 6809, 7128])
StromErr = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
RateErr = Rate/30



# xdata = np.asarray(data['Hochspannung U_H/kV'], dtype=np.float64)
# xerr = np.asarray(data['Fehler U_H/kV'], dtype=np.float64)
# ydata = np.asarray(data['Driftstrom /mikroA'], dtype= np.float64)
# yerr = np.asarray(data['Fehler Driftstrom /mikroA'], dtype=np.float64)


# h = 'driftstrom_mit.csv'
 



# def fit_odr(Para, x):
#     return Para[0]*np.exp(Para[1]*(x))+Para[2]

# linear = odr.Model(fit_odr)
# mydata = odr.RealData(xdata, ydata, sx= xerr, sy= yerr)
# myodr = odr.ODR(mydata, linear, beta0=[1e-8, 12., 7], maxit=20)
# out = myodr.run()


# residuals = ydata - fit_odr(out.beta, xdata)
# chisq_odr = np.sum((residuals**2)/yerr**2)
# print ('$\chi^2$=', chisq_odr, '$\chi$/ndf=', chisq_odr/(len(xdata)-len(out.beta)))
# fig, ax= plt.subplots()


# fig, ax= plt.subplots()
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((y-np.mean(y))**2)
# rsquared = 1 - (ss_res / ss_tot)





plt.grid()
plt.errorbar(Strom, Rate, xerr=StromErr, yerr= RateErr, color='midnightblue',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none')
# ax.set(ylabel='Driftstrom $I_{Drift}$ /$\mu$A', xlabel='Hochspannung $U_{Hoch}$ /kV')
# ax.set_ylim(-50, 1200)
# ax.legend()
plt.show