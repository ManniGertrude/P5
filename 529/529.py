import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd
from sklearn.metrics import r2_score



def read_csv_input(test):
    with open(test, "r") as file:
        dictR = csv.DictReader(file, delimiter=",")
        data = {hdr: [] for hdr in dictR.fieldnames}
        for row in dictR:
            for field in row:
                try:
                    data[field].append(float(row[field]))
                except ValueError:
                    data[field].append(0.)
    return data

input_dir = "./"
f = "stabdosimeter.csv"

data = read_csv_input(os.path.join(input_dir, f))

xmo = np.asarray(data['r/cm'], dtype=np.float64)
xerrmo = np.asarray(data['fehler r/cm'], dtype=np.float64)
ymo = np.asarray(data['D/mSvmin Mo'], dtype= np.float64)
yerrmo = np.asarray(data['Fehler mSvmin Mo'], dtype=np.float64)

xcu = np.asarray(data['r/cm'], dtype=np.float64)
xerrcu = np.asarray(data['fehler r/cm'], dtype=np.float64)
ycu = np.asarray(data['D/mSvmin Cu'], dtype= np.float64)
yerrcu = np.asarray(data['Fehler mSvmin Cu'], dtype=np.float64)



r = np.asarray(data['r/cm'], dtype=np.float64)
rerr = np.asarray(data['fehler r/cm'], dtype=np.float64)
dt_mo = np.asarray(data['D/t mSv/s Mo'], dtype=np.float64)
dt_mo_err = np.asarray(data['Fehler D/t Mo'], dtype=np.float64)
dt_cu = np.asarray(data['D/t mSv/s Cu'], dtype=np.float64)
dt_cu_err = np.asarray(data['Fehler D/t Cu'], dtype=np.float64)



x = np.arange(10, 40, 0.1)

def abstand(x, a, c):
    return a/(x**2 ) + c


p0mo= ([100000, 3])
popt, pcov = curve_fit(abstand, xmo, ymo, sigma= yerrmo, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)


ymobar = (np.sum(ymo)/len(ymo))
rsquaredmo = r2_score(ymo, abstand(xmo, *popt))
print ('$R^2$ der Molybdänröhre:', rsquaredmo)
#print ('Molybdänröhre Werte:', xmo,ymo)
print ('Molybdänröhre Parameter:', popt, perr)

p0c = ([5000,  -3])
poptc, pcovc = curve_fit(abstand, xcu, ycu,p0= p0c,  sigma= yerrcu, absolute_sigma=True)
perrc = np.sqrt(np.diag(pcovc))
covc= np.linalg.cond(pcovc)
dcovc = np.diag(pcovc)


ycubar = (np.sum(ycu)/len(ycu))
rsquaredcu = r2_score(ycu, abstand(xcu, *poptc))
print ('$R^2$ der Kupferröhre:', rsquaredcu)
#print ('Kupferröhre Werte:', xcu,ycu)
print ('Kupferröhre Parameter:', poptc, perrc)


fig, ax= plt.subplots()

plt.grid()
#plt.bar(xdata1, ydata1, width=0.1, bottom=None, color ='indianred',edgecolor= 'firebrick')
plt.errorbar(xmo, ymo, xerr=xerrmo, yerr= yerrmo, color='purple', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Molybdänröhre')
plt.plot(x, abstand(x, *popt), color ='orchid', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Molybdänröhre')
plt.errorbar(xcu, ycu, xerr=xerrcu, yerr= yerrcu, color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Kupferröhre')
plt.plot(x, abstand(x, *poptc), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Kupferröhre')
#plt.bar(u, i, width=0.1, bottom=None, color ='cornflowerblue',edgecolor= 'slateblue')
#plt.errorbar(xdata, ydata, xerr=xerr, yerr= yerr, color='midnightblue', marker='+', capsize=2, linestyle='none', label= 'Datenpunkte ohne $^{90}$Sr-Quelle')
#plt.plot(x, fit(x, *popt), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ohne $^{90}$Sr-Quelle')
ax.set(ylabel=' h$_{min}$ /mSvmin$^{-1}$', xlabel='Abstand r/cm')

ax.legend()

fig.savefig('abstand.png')
fig.savefig('abstand.pdf')
plt.show

# ohne ersten wert beider reihen

xmo_fil = xmo[1:10]
xerrmo_fil = xerrmo[1:10]
ymo_fil = ymo[1:10]
yerrmo_fil = yerrmo[1:10]

xcu_fil = xcu[1:10]
xerrcu_fil = xerrcu[1:10]
ycu_fil = ycu[1:10]
yerrcu_fil = yerrcu[1:10]

p0mo= ([100000, 3])
popt_mofil, pcov_mofil = curve_fit(abstand, xmo_fil, ymo_fil, sigma= yerrmo_fil, absolute_sigma=True)
perr_mofil = np.sqrt(np.diag(pcov_mofil))
cov_mofil= np.linalg.cond(pcov_mofil)
dcov_mofil = np.diag(pcov_mofil)


ymobarfil = (np.sum(ymo_fil)/len(ymo_fil))
rsquaredmofil = r2_score(ymo_fil, abstand(xmo_fil, *popt_mofil))
print ('$R^2$ der Molybdänröhre gefiltert:', rsquaredmofil)
#print ('Molybdänröhre Werte:', xmo_fil,ymo_fil)
print ('Molybdänröhre Parameter:', popt_mofil, perr_mofil)

p0c = ([5000,  -3])
poptcfil, pcovcfil = curve_fit(abstand, xcu_fil, ycu_fil,p0= p0c,  sigma= yerrcu_fil, absolute_sigma=True)
perrcfil = np.sqrt(np.diag(pcovcfil))
covcfil= np.linalg.cond(pcovcfil)
dcovcfil = np.diag(pcovcfil)


ycubarfil = (np.sum(ycu_fil)/len(ycu_fil))
rsquaredcufil = r2_score(ycu_fil, abstand(xcu_fil, *poptcfil))
print ('$R^2$ der Kupferröhre gefiltert:', rsquaredcufil)
#print ('Kupferröhre Werte:', xcu_fil,ycu_fil)
print ('Kupferröhre Parameter:', poptcfil, perrcfil)

fig, ax= plt.subplots()

plt.grid()

plt.errorbar(xmo_fil, ymo_fil, xerr=xerrmo_fil, yerr= yerrmo_fil, color='purple', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Molybdänröhre')
plt.plot(x, abstand(x, *popt_mofil), color ='orchid', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Molybdänröhre')
plt.errorbar(xcu_fil, ycu_fil, xerr=xerrcu_fil, yerr= yerrcu_fil, color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Kupferröhre')
plt.plot(x, abstand(x, *poptcfil), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Kupferröhre')
ax.set(ylabel='h$_{min}$ /mSvmin$^{-1}$', xlabel='Abstand r/cm')

ax.legend()

fig.savefig('abstand_fil.png')
fig.savefig('abstand_fil.pdf')
plt.show


#abstand für dosis pro sekunde 

rfil = r[1:10]
rerrfil = rerr[1:10]
dt_mofil = dt_mo[1:10]
dt_mo_errfil = dt_mo_err[1:10]
dt_cufil = dt_cu[1:10]
dt_cu_errfil = dt_cu_err[1:10]

print('gefilterte Abstände:', rfil)

optcu, covcu = curve_fit(abstand, r, dt_cu, sigma= dt_cu_err, absolute_sigma=True)
errcu = np.sqrt(np.diag(covcu))
rsquaredcudt = r2_score(dt_cu, abstand(r, *optcu))
print ('$R^2$ der Kupferröhre pro sekunde:', rsquaredcudt)
print('Parameter und Fehler für Kupferdosis pro Sekunde:', optcu, errcu)



optmo, covmo = curve_fit(abstand, r, dt_mo, sigma= dt_mo_err, absolute_sigma=True)
errmo = np.sqrt(np.diag(covmo))
rsquaredmodt = r2_score(dt_mo, abstand(r, *optmo))
print ('$R^2$ der Molybdänröhre pro sekunde:', rsquaredmodt)
print('Parameter und Fehler für Molybdändosis pro Sekunde:', optmo, errmo)



optcufil, covcufil = curve_fit(abstand, rfil, dt_cufil, sigma= dt_cu_errfil, absolute_sigma=True)
errcufil = np.sqrt(np.diag(covcufil))
rsquaredcudtfil = r2_score(dt_cufil, abstand(rfil, *optcufil))
print ('$R^2$ der Kupferröhre pro sekunde gefiltert:', rsquaredcudtfil)
print('Parameter und Fehler für Kupferdosis pro Sekunde:', optcufil, errcufil)



optmofil, covmofil = curve_fit(abstand, rfil, dt_mofil, sigma= dt_mo_errfil, absolute_sigma=True)
errmofil = np.sqrt(np.diag(covmofil))
rsquaredmodtfil = r2_score(dt_mofil, abstand(rfil, *optmofil))
print ('$R^2$ der Molybdänröhre pro sekunde gefiltert:', rsquaredmodtfil)
print('Parameter und Fehler für Molybdändosis pro Sekunde:', optmofil, errmofil)


fig, ax= plt.subplots()

plt.grid()

plt.errorbar(r, dt_mo, xerr=rerr, yerr= dt_mo_err, color='purple', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Molybdänröhre')
plt.plot(x, abstand(x, *optmo), color ='orchid', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Molybdänröhre')
plt.errorbar(r, dt_cu, xerr=rerr, yerr= dt_cu_err, color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Kupferröhre')
plt.plot(x, abstand(x, *optcu), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Kupferröhre')
ax.set(ylabel=' h$_{s}$ /mSvs$^{-1}$', xlabel='Abstand r/cm')

ax.legend()

fig.savefig('abstandprosec.png')
fig.savefig('abstandprosec.pdf')
plt.show

fig, ax= plt.subplots()

plt.grid()

plt.errorbar(rfil, dt_mofil, xerr=rerrfil, yerr= dt_mo_errfil, color='purple', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Molybdänröhre')
plt.plot(x, abstand(x, *optmofil), color ='orchid', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Molybdänröhre')
plt.errorbar(rfil, dt_cufil, xerr=rerrfil, yerr= dt_cu_errfil, color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte für Kupferröhre')
plt.plot(x, abstand(x, *optcufil), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für Kupferröhre')
ax.set(ylabel=' h$_{s}$ /mSvs$^{-1}$', xlabel='Abstand r/cm')

ax.legend()

fig.savefig('abstandprosecfil.png')
fig.savefig('abstandprosecfil.pdf')
plt.show












input_dir = "./"
g = "totzeit.csv"

werte = read_csv_input(os.path.join(input_dir, g))


#geiger-müller-zählrohr winkelmessung

sensor = np.asarray(werte['grad sensor'], dtype=np.float64)
sensorerr = np.asarray(werte['fehler grad'], dtype=np.float64)
rate = np.asarray(werte['zahlrate sensor'], dtype=np.float64)
rateerr = np.asarray(werte['fehler zahlrate sensor'], dtype=np.float64)

x = np.arange(-3, 3, 0.1)
fig, ax= plt.subplots()

plt.grid()

plt.errorbar(sensor, rate, xerr=sensorerr, yerr= rateerr, color='darkolivegreen', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte')
ax.set_yscale('log')
ax.set(ylabel='gemittelte Zählrate über $\Delta$t=30s', xlabel='Winkel Geiger-Müller-Zählrohr')

ax.legend()

fig.savefig('winkelsensor.png')
fig.savefig('winkelsensor.pdf')
plt.show


#totzeitmessung

#def para(x, t, ko):
 #   return ko*x/(1 + t*x)

#def nichtpara(x, tau, kor):
 #   return kor*x*np.exp(-x*tau)

#def lin(x, l, k):
#    return l*x + k

#Ie= np.asarray(werte['I/ mA'], dtype=np.float64)
#Ieerr = np.asarray(werte['fehler I/mA'], dtype=np.float64)
#N = np.asarray(werte['Zahlrate/s'], dtype=np.float64)
#Nerr = np.asarray(werte['fehler Zahlrate/s'], dtype=np.float64)


#p0tau= ([0.0001, 5])
#optpara, covpara = curve_fit(para, Ie, N, p0=p0tau, sigma=Nerr, absolute_sigma=True)
#errpara = np.sqrt(np.diag(covpara))
#para_chisq = np.sum(((N-para(Ie, *optpara))**2)/Nerr**2)
#para_chisqndf = para_chisq/(len(Ie)- len(optpara))
#print('Parameter des PAralysefit:', optpara,errpara,'Chi Quadrat des Paralysefit:', para_chisq, para_chisqndf)

#optnpara, covnpara = curve_fit(nichtpara, Ie, N, p0= p0tau, sigma=Nerr, absolute_sigma=True)
#errnpara = np.sqrt(np.diag(covnpara))
#npara_chisq = np.sum(((N-para(Ie, *optnpara))**2)/Nerr**2)
#npara_chisqndf = npara_chisq/(len(Ie)- len(optnpara))
#print('Parameter des PAralysefit:', optnpara,errnpara,'Chi Quadrat des Paralysefit:', npara_chisq, npara_chisqndf)

#Ielin = Ie[0:7]
#Ilinerr = Ieerr[0:7]
#Nlin = N[0:7]
#Nerrlin = Nerr[0:7]
#print ('Nerr:', Nerr)

#p0lin= ([1000000, 0.001])
#poptlin, pcovlin = curve_fit(lin,Ielin, Nlin, sigma= Nerrlin, absolute_sigma=True)
#perrlin = np.sqrt(np.diag(pcovlin))

#mlin = lin(Ielin, *poptlin)
#chisqtot = np.sum(((Nlin-mlin)**2)/Nerrlin**2)
#chindftot= chisqtot/(len(Ielin)- len(poptlin))
#print ('$\chi^2$=', chisqtot, '$\chi$/ndf=', chindftot)
#print ('lineare Zählrate Werte:', Ielin,Nlin)
#print ('Fehler:', Ilinerr, Nerrlin)
#print ('lineare Zählrate Parameter:', poptlin, perrlin)

#m= lin(Ie, *poptlin)

#fehler m ist wurzel des residuums
#merr = (np.abs(N- m))**(1/2)
#print (merr)

#paraopt, paracov = curve_fit(para, m, N, sigma= Nerr, absolute_sigma=True)
#paraerr = np.sqrt(np.diag(paracov))
#print('paralyse parameter:', paraopt,paraerr)

#nicht0= ([0.0000004, 1])
#nparaopt, nparacov = curve_fit(nichtpara, m, N, p0= nicht0, sigma= Nerr, absolute_sigma=True)
#nparaerr = np.sqrt(np.diag(nparacov))
#print('nicht paralyse parameter:', nparaopt, nparaerr)

#x = np.arange(0, 1, 0.05)

#fig, ax= plt.subplots()
#plt.grid()
#plt.errorbar(Ielin, Nlin, xerr=Ilinerr, yerr= Nerrlin, color='darkolivegreen', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte')
#plt.plot(Ielin, lin(Ielin, *poptlin),color ='yellowgreen', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für lineare Zählrate')
#ax.set(ylabel='gemittelte Zählrate über $\Delta$t=30s', xlabel='Emissionstrom $I_E$/mA')
#ax.legend()

#fig.savefig('totzeitlin.png')
#fig.savefig('totzeitlin.pdf')
#plt.show

#fig, ax= plt.subplots()
#plt.grid()
#plt.errorbar(Ie, N, xerr=Ieerr, yerr= Nerr, color='darkolivegreen', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte')
#plt.plot(x, lin(x, *poptlin),color ='yellowgreen', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für lineare Zählrate')
#plt.plot(x, para(x, *optpara),color ='slateblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für paralyse')
#plt.plot(x, nichtpara(x, *optnpara),color ='midnightblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für nicht paralyse')
#ax.set(ylabel='gemittelte Zählrate über $\Delta$t=30s', xlabel='Emissionstrom $I_E$/mA')
#ax.legend()

#fig.savefig('totzeit.png')
#fig.savefig('totzeit.pdf')
#plt.show

#fig, ax= plt.subplots()
#plt.grid()
#plt.errorbar(m, N, xerr=merr, yerr= Nerr, color='darkolivegreen', marker='+',capsize=2,  linestyle='none', label= 'Datenpunkte')
#plt.plot(m, para(m, *paraopt),color ='slateblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für paralyse')
#plt.plot(m, nichtpara(m, *nparaopt),color ='midnightblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion für nicht paralyse')
#ax.set(ylabel='gemessene Zählrate m', xlabel='estimierte Rate n')
#ax.legend()

#fig.savefig('totzeitmn.png')
#fig.savefig('totzeitmn.pdf')
#plt.show