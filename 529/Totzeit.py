import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np
from sklearn.metrics import r2_score



Strom = np.array([0.0001, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
StromErr = Strom/10
Rate = np.array([1, 1174, 1749, 1979, 2161, 2309, 2449, 2578, 3110, 3495, 3828, 4115, 4647, 5192, 5607, 6109, 6470, 6809, 7128])
RateErr = Rate/(10*np.sqrt(30))

fig, ax= plt.subplots()

# Linearer Fit
 
def lin_fit(Para, x):
     return Para[0] * x + Para[1]

lin_model = odr.Model(lin_fit)
lin_mydata = odr.RealData(Strom[0:5], Rate[0:5], sx= StromErr[0:5], sy= RateErr[0:5])
lin_myodr = odr.ODR(lin_mydata, lin_model, beta0=[1, 2])
lin_out = lin_myodr.run()

lin_y = lin_fit(lin_out.beta, Strom)

lin_residuals = Rate - lin_y
lin_chisq_odr = np.sum((lin_residuals**2)/Rate**2)
lin_rsquared = r2_score(Rate[0:5], lin_y[0:5])
# lin_out.pprint()
print('$Parameter:', lin_out.beta, lin_out.sd_beta,  '$')
print ('$\chi_{lin}^2 =', lin_chisq_odr, '\chi/ndf =', lin_chisq_odr/(len(Strom)-len(lin_out.beta)), '$')
print('$R_{lin}^2 =',lin_rsquared, '$')
print()

plt.plot(Strom, lin_y, c="red", label='Linearer Fit')





# Nicht Paralysiert

def no_paraly_fit(Para, x):
    n = lin_fit(lin_out.beta, x)
    return n/(n*Para+1)

np_model = odr.Model(no_paraly_fit)
np_mydata = odr.RealData(Strom, Rate, sx= StromErr, sy= RateErr)
np_myodr = odr.ODR(np_mydata, np_model, beta0=[0.00005], maxit=1000)
np_out = np_myodr.run()
np_y = no_paraly_fit(np_out.beta, Strom)

np_residuals = Rate - np_y
np_chisq_odr = np.sum((np_residuals**2)/Rate**2)
np_rsquared = r2_score(Rate, np_y)
# np_out.pprint()
print('$Parameter:', np_out.beta,  np_out.sd_beta, '$')
print ('$\chi_{np}^2 =', np_chisq_odr, '\chi/ndf =', np_chisq_odr/(len(Strom)-len(np_out.beta)), '$')
print('$R_{np}^2 =',np_rsquared, '$')
print()
plt.plot(Strom, np_y, c="blue", label='nicht paralysierter Fit')




# Paralysiert

def yes_paraly_fit(Para, x):
    n = lin_fit(lin_out.beta, x)
    return  n* np.exp(-n*Para)

yp_model = odr.Model(yes_paraly_fit)
yp_mydata = odr.RealData(Strom, Rate, sx= StromErr, sy= RateErr)
yp_myodr = odr.ODR(yp_mydata, yp_model, beta0=[1e-5], maxit=1000)
yp_out = yp_myodr.run()
yp_y =  yes_paraly_fit(yp_out.beta, Strom) #lin_y* np.exp(-lin_y*0.000029)/1.8     #

yp_residuals = Rate - yp_y
yp_chisq_odr = np.sum((yp_residuals**2)/Rate**2)
yp_rsquared = r2_score(Rate, yp_y)
# yp_out.pprint()
print('$Parameter:', yp_out.beta, yp_out.sd_beta,  '$')
print ('$\chi_{np}^2 =', yp_chisq_odr, '\chi/ndf =', yp_chisq_odr/(len(Strom)-len(yp_out.beta)), '$')
print('$R_{np}^2 =',yp_rsquared, '$')
print()
plt.plot(Strom, yp_y, c="green", label='paralysierter Fit')





# Beides

def quantum_paraly_fit(Para, x):
    n = lin_fit(lin_out.beta, x)
    return  (Para[1]*n* np.exp(-n*Para[0]) + (1-Para[1])*n/(n*Para[0]+1))

q_model = odr.Model(quantum_paraly_fit)
q_mydata = odr.RealData(Strom, Rate, sx= StromErr, sy= RateErr)
q_myodr = odr.ODR(q_mydata, q_model, beta0=[1e-5, 0.5], maxit=1000)
q_out = q_myodr.run()
q_y =  quantum_paraly_fit(q_out.beta, Strom)

q_residuals = Rate - q_y
q_chisq_odr = np.sum((q_residuals**2)/Rate**2)
q_rsquared = r2_score(Rate, q_y)
# q_out.pprint()
print('$Parameter:', q_out.beta, q_out.sd_beta,  '$')
print ('$\chi_{q}^2 =', q_chisq_odr, '\chi/ndf =', q_chisq_odr/(len(Strom)-len(q_out.beta)), '$')
print('$R_{q}^2 =',q_rsquared, '$')
plt.plot(Strom, q_y, c="gold", label='quantenparalysierter Fit')







plt.grid()
plt.errorbar(Strom, Rate, xerr=StromErr, yerr= RateErr, color='purple',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label='Messwerte')

# ax.set(ylabel='Driftstrom $I_{Drift}$ /$\mu$A', xlabel='Hochspannung $U_{Hoch}$ /kV')
ax.set_ylim(-50, 10000)
ax.legend()
plt.savefig("P5\\529\\Totzeit.pdf")
plt.show