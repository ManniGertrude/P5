import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np

Material = np.array(['Luft', 'Kohlenstoff', 'Aluminium', 'Eisen', 'Kupfer', 'Zirkonium', 'Silber'])
OZahl = np.array([0, 6, 13, 26, 29, 40, 47])
Dichte = np.array([0, 2.26, 2.6989, 7.874, 8.96, 6.501, 10.49 ])
Rate = np.array([2627, 2538, 2403, 1283, 227.1, 340.8, 97.27])
RateErr = Rate/(10*np.sqrt(30))
Zr_Rate = np.array([1200, 1185, 1060, 654.3, 113, 206.9, 20.27])
Zr_RateErr = Zr_Rate/(20*np.sqrt(30))
Strom = np.array([0.02, 0.02, 0.03, 1, 1, 1, 1])
StromErr = Strom/50


NormRate = Rate/(Strom)
NormRateErr = np.sqrt((RateErr/Strom)**2 + (Rate*StromErr/Strom**2)**2)

Zr_NormRate = Zr_Rate/(Strom)
Zr_NormRateErr = np.sqrt((Zr_RateErr/Strom)**2 + (Zr_Rate*StromErr/Strom**2)**2)

fig, ax= plt.subplots()
 

plt.grid()

plt.errorbar(Dichte, NormRate, xerr=0, yerr=NormRateErr, color='red',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label='Ohne Zr')
plt.errorbar(Dichte, Zr_NormRate, xerr=0, yerr=Zr_NormRateErr, color='blue',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label='Mit Zr')
ax.legend()
plt.yscale('linear')
plt.savefig("P5\\529\\Verschiedene Mats lin.pdf")
plt.yscale('log')
plt.savefig("P5\\529\\Verschiedene Mats log.pdf")
plt.show