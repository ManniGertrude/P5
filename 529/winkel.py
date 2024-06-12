import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np
from sklearn.metrics import r2_score

fig, ax= plt.subplots()

Winkel = np.array([3, 2.5, 2, 1.5, 1, 0.5, 1e-4, -0.2, -0.3, -0.4, -0.5, -1, -1.5, -2, -2.5, -3])
WinkelErr = 0.1
Rate = np.array([1e-4, 1e-4, 1e-4, 2, 300, 1420, 2200, 2300, 2320, 2300, 2250, 1630, 670, 75, 15, 3])
RateErr = Rate/(10*np.sqrt(30))+10

def lin_fit(Para, x):
    return Para[2]/(Para[0]*np.sqrt(2*np.pi)) * np.exp(-1/2*((x - Para[1])/Para[0])**2)
 
# def lin_fit(Para, x):
#      return Para[0] * np.cos(Para[2]*(x + Para[1]))**2


lin_model = odr.Model(lin_fit)
lin_mydata = odr.RealData(Winkel, Rate, sx=WinkelErr, sy=RateErr)
lin_myodr = odr.ODR(lin_mydata, lin_model, beta0=[1, 0.3, 2350], maxit=1000)
lin_out = lin_myodr.run()

x = np.linspace(-3, 3, 100)
y = lin_fit(lin_out.beta, x)
lin_y = lin_fit(lin_out.beta, Winkel)

lin_residuals = Rate - lin_y
lin_chisq_odr = np.sum((lin_residuals**2)/Rate**2)
lin_rsquared = r2_score(Rate, lin_y)
# lin_out.pprint()
print('$Parameter:', lin_out.beta, lin_out.sd_beta,  '$')
# print ('$\chi_{lin}^2 =', lin_chisq_odr, '\chi/ndf =', lin_chisq_odr/(len(Winkel)-len(lin_out.beta)), '$')
# print('$R_{lin}^2 =',lin_rsquared, '$')

plt.plot(x, y, c="red")





plt.errorbar(Winkel, Rate,xerr=WinkelErr, yerr=RateErr, color='blue',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none')
# ax.legend()
plt.grid()
plt.ylabel('Messrate in $s^{-1}$')
plt.xlabel('Winkeleingabe des Goniometers in $^\circ$')
plt.savefig("P5\\529\\Winkel.pdf")
plt.show