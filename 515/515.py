import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd



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
f = "driftstrom_ohne.csv"

data = read_csv_input(os.path.join(input_dir, f))

xdata = np.asarray(data['Hochspannung U_H/kV'], dtype=np.float64)
xerr = np.asarray(data['Fehler U_H/kV'], dtype=np.float64)
ydata = np.asarray(data['Driftstrom /mikroA'], dtype= np.float64)
yerr = np.asarray(data['Fehler Driftstrom /mikroA'], dtype=np.float64)


h = 'driftstrom_mit.csv'
data = read_csv_input(os.path.join(input_dir, h))

xdata1 = np.asarray(data['Hochspannung U_H/kV'], dtype=np.float64)
xerr1 = np.asarray(data['Fehler U_H/kV'], dtype=np.float64)
ydata1 = np.asarray(data['Driftstrom /mikroA'], dtype= np.float64)
yerr1 = np.asarray(data['Fehler Driftstrom /mikroA'], dtype=np.float64)

x = np.arange(1.4, 3, 0.1)
 
u = [1.5,  1.7,  1.9,  2.1,  2.3,  2.4,  2.5,  2.6,  2.65, 2.7,  2.75, 2.8,  2.85, 2.9,
 2.95]
i = [ 4,  2,  4,  6,  8,  8, 10, 14, 16, 18, 22, 32, 42, 54, 84]
#print (xdata)
#print (ydata)

def fit(x, a, b, c):
    return a*np.exp(b*x) + c

popt, pcov = curve_fit(fit, xdata, ydata, sigma= yerr, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)

#print ('Parameter und Fehler ohne Strahlungsquelle:', popt, cov, dcov, perr)

ydatapre = fit(xdata, *popt)
res = ydata - ydatapre
chisq = np.sum(((ydata-ydatapre)**2)/yerr**2)
#print ('Residuum ', res)
print ('$\chi^2$=', chisq, '$\chi$/ndf=', chisq/(15-3))
[out1, out2, out3] = popt

popt1, pcov1 = curve_fit(fit, xdata1, ydata1, sigma= yerr1, absolute_sigma=True)
perr1 = np.sqrt(np.diag(pcov1))
cov1= np.linalg.cond(pcov1)
dcov1 = np.diag(pcov1)

#print ('Parameter und Fehler mit Strahlungsquelle:', popt1, cov1, dcov1, perr1)
ydatapre1 = fit(xdata1, *popt1)
res1 = ydata1 - ydatapre1
chisq1 = np.sum(((ydata1-ydatapre1)**2)/yerr1**2)
#print ('Residuum ', res)
#print ('$\chi^2$=', chisq1, '$\chi$/ndf=', chisq1/(15-3))


def fit_odr(x, a):
    return a[0]*np.exp(a[1]*x) + a[2]


linear = odr.Model(fit_odr)
mydata = odr.RealData(xdata, ydata, sx= xerr, sy= yerr)
myodr = odr.ODR(mydata, linear, beta0=[1., 1., 1.])
out = myodr.run()
out.pprint()

residuals = ydata - fit_odr(x, out.beta)
chisq_odr = np.sum((residuals**2)/yerr**2)
print ('$\chi^2$=', chisq_odr, '$\chi$/ndf=', chisq_odr/(15-3))

fig, ax= plt.subplots()


plt.grid()
#plt.bar(xdata1, ydata1, width=0.1, bottom=None, color ='indianred',edgecolor= 'firebrick')
plt.errorbar(xdata1, ydata1, xerr=xerr1, yerr= yerr1, color='maroon', marker='+', linestyle='none', label= 'Datenpunkte mit $^{90}$Sr-Quelle')
plt.plot(x, fit(x, *popt1), color ='indianred', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion mit $^{90}$Sr-Quelle')
#plt.bar(u, i, width=0.1, bottom=None, color ='cornflowerblue',edgecolor= 'slateblue')
plt.errorbar(xdata, ydata, xerr=xerr, yerr= yerr, color='midnightblue', marker='+', linestyle='none', label= 'Datenpunkte ohne $^{90}$Sr-Quelle')
plt.plot(x, fit(x, *popt), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ohne $^{90}$Sr-Quelle')
ax.set(ylabel='Driftstrom $I_{Drift}$ /$\mu$A', xlabel='Hochspannung $U_{Hoch}$ /kV')

ax.legend()

fig.savefig('driftstromfit.png')
fig.savefig('driftstromfit.pdf')
plt.show