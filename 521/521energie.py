import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

data = np.loadtxt("521energie.txt", dtype=str)
#data = np.char.replace(data, ',', '.')
data = data.astype(np.float64)
array = np.split(data, 5, axis=1)

E= np.ravel(array[0], order='C')
#rechts
mu = np.ravel(array[1], order='C')
muerr = np.ravel(array[2], order='C')
fwhm = np.ravel(array[3], order='C')[3:]
fwhm_err = np.ravel(array[4], order='C')[3:]
x = np.linspace(0, 1400, 10000)

fig, ax=plt.subplots()

def func(x, a, b) :
    return a + b*x 

kanalerr = np.zeros(len(E))
for i in range (0, len(E)):
    kanalerr[i] = np.sqrt((muerr[i])**2)

popt,pcov= curve_fit(func, E, mu, sigma=kanalerr, absolute_sigma=True, maxfev=5000)

perr = np.sqrt(np.diag(pcov))
print (f'Ausgleichsgerade: $({popt[1]}\pm{perr[1]}) \cdot x + ({popt[0]}\pm{perr[0]})$')
print('Werte rechts:', mu, kanalerr)
rsquaredfil = r2_score(mu,func(E, *popt))
print (f'$R^2$ ', rsquaredfil)
plt.errorbar(E, mu, yerr=kanalerr,  color='cornflowerblue',linestyle ='none', marker ='.', markersize = 10, elinewidth= 5 ,label='Datenpunkte')
plt.plot(x, func(x, *popt), color = 'midnightblue',linestyle = '-', label =f'Kalibrierungsgerade')
plt.grid()

ax.set(ylabel='Kanalnummer ', xlabel='Energie E /keV ')
ax.legend()

fig.savefig(f"kalibrierung_Halbleiter.pdf")
plt.show


Energie= np.ravel(array[0], order='C')[3:]
deltae = fwhm**2
deltaerr = (2*fwhm*fwhm_err)**2
En= np.ravel(array[0], order='C')[-3:]
rest = np.ravel(array[3], order='C')[-3:]
resterr = np.ravel(array[4], order='C')[-3:]

print (Energie, fwhm, fwhm_err)
fig,ax = plt.subplots()
p0=[5, 0.0001]
popt_e, pcov_e = curve_fit(func, Energie, deltae,sigma=deltaerr, absolute_sigma=True, p0=p0)
perr_e = np.sqrt(np.diag(pcov_e))
print (f'Ausgleichsgerade: $({popt_e[1]}\pm{perr_e[1]}) \cdot FWHM^2  + ({popt_e[0]}\pm{perr_e[0]})$')
rsquaredfil = r2_score(deltae,func(Energie, *popt_e))
print (f'$R^2$ ', rsquaredfil)
plt.errorbar(Energie, fwhm**2, yerr=(2*fwhm*fwhm_err)**2,  color='cornflowerblue',linestyle ='none', marker ='.', markersize = 7, elinewidth= 2 ,label='Datenpunkte')
plt.errorbar(En, rest**2, yerr=(2*rest*resterr)**2, color ='indigo', linestyle ='none', marker = '.',markersize = 7, elinewidth= 2, label ='nicht in Anpassung verwendete Daten')
plt.plot(x, func(x, *popt_e), color = 'midnightblue',linestyle = '-', label =f'Kalibrierungsgerade')
plt.grid()

ax.set(ylabel='Quadrat der Halbwertsbreiten ', xlabel='Energie E /keV ')
ax.legend()

fig.savefig(f"kalibrierung_FWHM.pdf")
plt.show