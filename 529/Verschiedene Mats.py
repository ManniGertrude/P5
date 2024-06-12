import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np

Material = np.array(['Luft', 'C', 'Al', 'Fe', 'Cu', 'Zr', 'Ag'])
OZahl = np.array([0, 6, 13, 26, 29, 40, 47])
Dichte = np.array([0, 2.26, 2.6989, 7.874, 8.96, 6.501, 10.49 ])
Rate = np.array([2627, 2538, 2403, 1283, 227.1, 340.8, 97.27])
RateErr = Rate/(10*np.sqrt(30))
Zr_Rate = np.array([1200, 1185, 1060, 654.3, 113, 206.9, 20.27])
Zr_RateErr = Zr_Rate/(20*np.sqrt(30))
Strom = np.array([0.02, 0.02, 0.03, 1, 1, 1, 1])
StromErr = Strom/50
muAlu = 0.75514846 # mm
muAluErr = 0.0456824
Zr_muAlu = 0.76836382 
Zr_muAluErr = 0.0456824

NormRate = Rate/(Strom)
NormRateErr = np.sqrt((RateErr/Strom)**2 + (Rate*StromErr/Strom**2)**2)

Zr_NormRate = Zr_Rate/(Strom)
Zr_NormRateErr = np.sqrt((Zr_RateErr/Strom)**2 + (Zr_Rate*StromErr/Strom**2)**2)

TValue = NormRate/NormRate[0]
TValueErr = NormRateErr/NormRate[0]
Zr_TValue = Zr_NormRate/Zr_NormRate[0]
Zr_TValueErr = Zr_NormRateErr/Zr_NormRate[0]


mu = -np.log(TValue)*2
mu = mu/mu[2]*muAlu
muErr = -np.log(TValueErr)*2
muErr = np.sqrt((muErr/muErr[2]*muAlu)**2 + (mu/mu[2]*muAluErr)**2 )


Zr_mu = -np.log(TValue)*2
Zr_mu = Zr_mu/Zr_mu[2]*Zr_muAlu
Zr_muErr = -np.log(Zr_TValueErr)*2
Zr_muErr = np.sqrt((Zr_muErr/Zr_muErr[2]*Zr_muAlu)**2 + (Zr_mu/Zr_mu[2]*Zr_muAluErr)**2 )



fig, ax= plt.subplots()
plt.errorbar(Material, mu, yerr=muErr,marker='+',markersize=8, color='red',capsize=4, elinewidth=1.7, capthick=1.2, linestyle='none', label='Ohne Zr')
plt.errorbar(Material, Zr_mu, yerr=Zr_muErr,marker='+',markersize=8, color='blue',capsize=4, elinewidth=1.3, capthick=0.8, linestyle='none', label='Mit Zr')
ax.legend()

plt.grid()
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
plt.yscale('linear')
plt.ylabel('Abschwächungskoeffizient $\mu$ / $mm^{-1}$')
plt.savefig("P5\\529\\Verschiedene Mats lin.pdf")
plt.yscale('log')
plt.savefig("P5\\529\\Verschiedene Mats log.pdf")
plt.show