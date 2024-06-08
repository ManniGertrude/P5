import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np


Dicke = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
DickeErr = 0.05
Rate = np.array([1200, 1017, 683.9, 1026, 1091, 1039, 1089])
RateErr = Rate/(10*np.sqrt(30))
Strom = np.array([0.02, 0.03, 0.03, 0.07, 0.11, 0.13, 0.19])
StromErr = Strom/10

NormRate = Rate/(Strom)
NormRateErr = np.sqrt((RateErr/Strom)**2 + (Rate*StromErr/Strom**2)**2)


fig, ax= plt.subplots()
 
def fit(Para, x):
     return np.exp(-Para[0]*(x + Para[1]) + Para[2])

model = odr.Model(fit)
mydata = odr.RealData(Dicke,NormRate, sx=DickeErr, sy=NormRateErr)
myodr = odr.ODR(mydata, model, beta0=[0.0001, 1, 1], maxit=1000)
out = myodr.run()
print("para:",out.beta)

yWerte = fit(out.beta, Dicke)
residuals = Dicke - yWerte
chisq_odr = np.sum((residuals**2)/NormRate**2)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((yWerte-np.mean(yWerte))**2)
rsquared = 1 - (ss_res / ss_tot)
# print ('chi^2 =', lin_chisq_odr, 'chi/ndf =', lin_chisq_odr/(len(Strom)-len(lin_out.beta)))
# print('R^2 =',lin_rsquared)


x = np.linspace(np.min(Dicke)-0.2, np.max(Dicke)+0.2, 100)
y = np.exp(-out.beta[0]*(x+out.beta[1]) + out.beta[2])
plt.plot(x, y, c="red")

plt.grid()
plt.errorbar(Dicke, NormRate, xerr=DickeErr, yerr=NormRateErr, color='purple',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label='Messwerte')
ax.legend()
plt.yscale('linear')
plt.savefig("P5\\529\\Verschiedene Dicke lin.pdf")
plt.yscale('log')
plt.savefig("P5\\529\\Verschiedene Dicke log.pdf")
plt.show