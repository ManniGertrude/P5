import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

data = np.loadtxt("C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\521\\521energie.txt", dtype=str)
data = data.astype(np.float64)
array = np.split(data, 5, axis=1)

E= np.ravel(array[0], order='C')
fwhm = np.ravel(array[3], order='C')**2 
fwhm_err = 2*np.ravel(array[4], order='C')*fwhm
x = np.linspace(0, 1400, 1000)

fig, ax=plt.subplots()

def func(x, a, b) :
    return a + b*x 

popt,pcov= curve_fit(func, E[:-3], fwhm[:-3], sigma=fwhm_err[:-3], absolute_sigma=True, maxfev=5000) 

perr = np.sqrt(np.diag(pcov))
print (f'Ausgleichsgerade: $({popt[1]}\pm{perr[1]}) \cdot x + ({popt[0]}\pm{perr[0]})$')
rsquaredfil = r2_score(fwhm,func(E, *popt))
print (f'$R^2$ ', rsquaredfil)
plt.errorbar(E, fwhm, yerr=fwhm_err,  color='firebrick',linestyle ='none', marker ='.', capsize=3, markersize = 4, elinewidth= 1.5 ,label='Datenpunkte')
plt.plot(x, func(x, *popt), color = 'midnightblue',linestyle = '-', label =f'Kalibrierungsgerade')
plt.grid()

ax.set(ylabel='Kanalnummer ', xlabel='Energie E /keV ')
ax.legend()

fig.savefig(f"kalibrierung_FWHM.png")
plt.show
