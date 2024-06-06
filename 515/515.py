import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os


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

input_dir = "C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\p5\\515\\"
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

x = np.linspace(1.4, 3, 100)
 

def fit(x, a, b, c):
    return a*np.exp(b*(x)) + c

popt, pcov = curve_fit(fit, xdata, ydata, sigma= yerr, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)


def fit_odr(Para, x):
    return Para[0]*np.exp(Para[1]*(x))+Para[2]

linear = odr.Model(fit_odr)
mydata = odr.RealData(xdata, ydata, sx= xerr, sy= yerr)
myodr = odr.ODR(mydata, linear, beta0=[1e-8, 12., 7], maxit=20)
out = myodr.run()
# out.pprint()

residuals = ydata - fit_odr(out.beta, xdata)
chisq_odr = np.sum((residuals**2)/yerr**2)
# print ('$\chi^2$=', chisq_odr, '$\chi$/ndf=', chisq_odr/(len(xdata)-len(out.beta)))
fig, ax= plt.subplots()

y = fit_odr(out.beta, x)


linear = odr.Model(fit_odr)
mydata1 = odr.RealData(xdata1, ydata1, sx= xerr1, sy= yerr1)
myodr1 = odr.ODR(mydata1, linear, beta0=[1., 1., 1.])
out1 = myodr1.run()

residuals1 = ydata1 - fit_odr(out1.beta, xdata1)
chisq_odr1 = np.sum((residuals1**2)/yerr1**2)
# print ('$\chi^2$=', chisq_odr1, '$\chi$/ndf=', chisq_odr1/(len(xdata1)-len(out1.beta)))
fig, ax= plt.subplots()
x1 = np.linspace(1.4, 2.9, 100)
y1 = fit_odr(out1.beta, x)


ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
rsquared = 1 - (ss_res / ss_tot)

ss_res1 = np.sum(residuals1**2)
ss_tot1 = np.sum((y1-np.mean(y1))**2)
rsquared1 = 1 - (ss_res1 / ss_tot1)

print(rsquared)
print(rsquared1)


def fitE(x, a, b, c):
    return a*np.exp(b*(x+0.03)) + c

XWerte = np.linspace(1.4, 2.9, 100)
YWerte = fitE(XWerte, out1.beta[0], out1.beta[1], out1.beta[2])


plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr= yerr, color='midnightblue',capsize=2, elinewidth=1.3, capthick=0.8, linestyle='none', label= 'Datenpunkte ohne $^{90}$Sr-Quelle')
plt.errorbar(xdata1, ydata1, xerr=xerr1, yerr= yerr1, color='darkolivegreen',capthick=0.8,elinewidth=1.3, capsize=2, linestyle='none', label= 'Datenpunkte mit $^{90}$Sr-Quelle')
plt.plot(x, y, color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ohne $^{90}$Sr-Quelle')
plt.plot(XWerte, YWerte, color ='yellowgreen', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion mit $^{90}$Sr-Quelle')
ax.set(ylabel='Driftstrom $I_{Drift}$ /$\mu$A', xlabel='Hochspannung $U_{Hoch}$ /kV')
ax.set_ylim(-50, 1200)
ax.legend()

fig.savefig('driftstromfit.png')
fig.savefig('driftstromfit.pdf')
plt.show