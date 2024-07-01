import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

data = np.loadtxt("energiekali.txt", dtype=str)
#data = np.char.replace(data, ',', '.')
data = data.astype(np.float64)
array = np.split(data, 9, axis=1)

E= np.ravel(array[0], order='C')
#rechts
mur = np.ravel(array[1], order='C')
murerr = np.ravel(array[2], order='C')
sigr= np.ravel(array[3], order='C')
sigrerr = np.ravel(array[4], order='C')
#links
mul = np.ravel(array[5], order='C')
mulerr = np.ravel(array[6], order='C')
sigl= np.ravel(array[7], order='C')
siglerr = np.ravel(array[8], order='C')

x = np.linspace(0, 600, 1000)

fig, ax=plt.subplots()

def func(x, a, b) :
    return a + b*x 

kanalrerr = np.zeros(len(E))
for i in range (0, len(E)):
    kanalrerr[i] = np.sqrt((murerr[i])**2 + sigr[i]**2 + (sigrerr[i])**2)

poptr,pcovr = curve_fit(func, E, mur, maxfev=2000)

perr = np.sqrt(np.diag(pcovr))
print (f'Parameter', poptr, perr)
print('Werte rechts:', mur, kanalrerr)
rsquaredfil = r2_score(mur,func(E, *poptr))
print (f'$R^2$ ', rsquaredfil)
plt.errorbar(E, mur, yerr=kanalrerr,  color='cornflowerblue',linestyle ='none', marker ='.', markersize = 10, elinewidth= 5 ,label='Datenpunkte')
plt.plot(x, func(x, *poptr), color = 'midnightblue',linestyle = '-', label =f'Kalibrierungsgerade')
plt.grid()

ax.set(ylabel='Kanalnummer ', xlabel='Energie E /keV ')
ax.legend()

fig.savefig(f"kalibrierungrechts.pdf")
plt.show

kanallerr = np.zeros(len(E))
for i in range (0, len(E)):
    kanallerr[i] = np.sqrt((mulerr[i])**2 + sigl[i]**2 + (siglerr[i])**2)

poptl,pcovl = curve_fit(func, E, mul, maxfev=2000)

perrl = np.sqrt(np.diag(pcovl))
print (f'Parameter', poptl, perrl)
print ('werte links:', mul, kanallerr)
rsquaredfill = r2_score(mul,func(E, *poptl))
print (f'$R^2$ ', rsquaredfill)

fig, ax = plt.subplots()
plt.errorbar(E, mul, yerr=kanallerr,  color='cornflowerblue',linestyle ='none', marker ='.', markersize = 10, elinewidth= 3 ,label='Datenpunkte')
plt.plot(x, func(x, *poptl), color = 'midnightblue',linestyle = '-', label =f'Kalibrierungsgerade')
plt.grid()

ax.set(ylabel='Kanalnummer ', xlabel='Energie E /keV ')
ax.legend()

fig.savefig(f"kalibrierunglinks.pdf")
plt.show

mlinks = poptl[1]
mlierr = perrl[1]
mrechts = poptr[1]
mrechtserr = perr[1]

print (mlinks, mlierr,mrechts, mrechtserr)
aufloesungrechts = np.zeros(len(E))
aufloesunglinks = np.zeros(len(E))
for i in range (0, len(E)):
   aufloesungrechts[i] = (2*np.sqrt(2*np.log(2))*sigr[i])/mrechts
   aufloesunglinks[i] = (2*np.sqrt(2*np.log(2))*sigl[i])/mlinks


print(aufloesunglinks, aufloesungrechts)
aufloesungrechtserr = np.zeros(len(E))
aufloesunglinkserr = np.zeros(len(E))
for i in range (0, len(E)):
   aufloesungrechtserr[i] = np.sqrt(((2*np.sqrt(2*np.log(2))*sigrerr[i])/mrechts)**2 + ((2*np.sqrt(2*np.log(2))*sigr[i]* mrechtserr)/mrechts**2)**2)
   aufloesunglinkserr[i] = np.sqrt(((2*np.sqrt(2*np.log(2))*siglerr[i])/mlinks)**2 + ((2*np.sqrt(2*np.log(2))*sigl[i]* mlierr)/mlinks**2)**2)

print(aufloesunglinkserr, aufloesungrechtserr)