import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np



Strom = np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
StromErr = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
Rate = np.array([1174, 1749, 1979, 2161, 2309, 2449, 2578, 3110, 3495, 3828, 4115, 4647, 5192, 5607, 6109, 6470, 6809, 7128])
RateErr = Rate/30

fig, ax= plt.subplots()


# Linearer Fit
 
def lin_fit(Para, x):
     return Para[0] * x + Para[1]

lin_model = odr.Model(lin_fit)
lin_mydata = odr.RealData(Strom[0:6], Rate[0:6], sx= StromErr[0:6], sy= RateErr[0:6])
lin_myodr = odr.ODR(lin_mydata, lin_model, beta0=[1, 2])
lin_out = lin_myodr.run()
print(lin_out.beta)

lin_y = lin_fit(lin_out.beta, Strom)

lin_residuals = Strom - lin_y
lin_chisq_odr = np.sum((lin_residuals**2)/Rate**2)
lin_ss_res = np.sum(lin_residuals**2)
lin_ss_tot = np.sum((lin_y-np.mean(lin_y))**2)
lin_rsquared = 1 - (lin_ss_res / lin_ss_tot)
# print ('chi^2 =', lin_chisq_odr, 'chi/ndf =', lin_chisq_odr/(len(Strom)-len(lin_out.beta)))
# print('R^2 =',lin_rsquared)
plt.plot(Strom, lin_y, c="red", label='Linearer Fit')





# Nicht Paralysiert

def no_paraly_fit(Para, x):
    n = lin_fit(lin_out.beta, x)
    return n/(n*Para+1)

np_model = odr.Model(no_paraly_fit)
np_mydata = odr.RealData(Strom, Rate, sx= StromErr, sy= RateErr)
np_myodr = odr.ODR(np_mydata, np_model, beta0=[0.001], maxit=1000)
np_out = np_myodr.run()
print(np_out.beta)
np_y = no_paraly_fit(np_out.beta, Strom)

np_residuals = Strom - np_y
np_chisq_odr = np.sum((np_residuals**2)/Rate**2)
np_ss_res = np.sum(np_residuals**2)
np_ss_tot = np.sum((np_y-np.mean(np_y))**2)
np_rsquared = 1 - (np_ss_res / np_ss_tot)
# print ('chi^2 =', np_chisq_odr, 'chi/ndf =', np_chisq_odr/(len(Strom)-len(np_out.beta)))
# print('R^2 =',np_rsquared)
plt.plot(Strom, np_y, c="blue", label='nicht paralysierter Fit')





# Paralysiert

def yes_paraly_fit(Para, x):
    n = lin_fit(lin_out.beta, x)
    return  n * np.exp(-n*Para[0] + Para[1])+Para[2]

yp_model = odr.Model(yes_paraly_fit)
yp_mydata = odr.RealData(Strom, Rate, sx= StromErr, sy= RateErr)
yp_myodr = odr.ODR(yp_mydata, yp_model, beta0=[0.000001, 0.1, 0.1], maxit=1000)
yp_out = yp_myodr.run()
print(yp_out.beta)
yp_y = yes_paraly_fit(yp_out.beta, Strom)

yp_residuals = Strom - yp_y
yp_chisq_odr = np.sum((yp_residuals**2)/Rate**2)
yp_ss_res = np.sum(yp_residuals**2)
yp_ss_tot = np.sum((yp_y-np.mean(yp_y))**2)
yp_rsquared = 1 - (yp_ss_res / yp_ss_tot)
# print ('chi^2 =', np_chisq_odr, 'chi/ndf =', np_chisq_odr/(len(Strom)-len(np_out.beta)))
# print('R^2 =',np_rsquared)
plt.plot(Strom, yp_y, c="green", label='paralysierter Fit')











plt.grid()
plt.errorbar(Strom, Rate, xerr=StromErr, yerr= RateErr, color='purple',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label='Messwerte')

# ax.set(ylabel='Driftstrom $I_{Drift}$ /$\mu$A', xlabel='Hochspannung $U_{Hoch}$ /kV')
ax.set_ylim(-50, 10000)
ax.legend()
plt.savefig("Totzeit.pdf")
plt.show