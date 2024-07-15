import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit 
import os
import scipy.odr as odr
import math
from sklearn.metrics import r2_score

Path = 'C:\\Users\\kontr\\Desktop\\Github\\P5\\521\\'
Path2 = 'C:\\Users\\kontr\\Desktop\\Github\\P5\\521\\Messungen\\MCA\\'

Faktor = 1/9.366075
udata = pd.read_csv(f'{Path2}LangzeitUntergrund.txt', sep=' ',header=0, names=['x', 'y'])
bdata = pd.read_csv(f'{Path2}bodenprobe.txt', sep=' ',header=0, names=['x', 'y'])
xData = bdata['x'].values[:-1]
FakeyData1 = np.array(bdata['y'].values)[:-1]
FakeyData2 = np.array(udata['y'].values)[:-1]
yData = FakeyData1 - FakeyData2


def gaus(Para, x): 
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2) + Para[3]*x + Para[4]


def gausfit(func, x, y, farbe, beta, Name):
    model = odr.Model(func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, model, beta0=beta)
    out = myodr.run()
    fy = func(out.beta, x)
    plt.plot(x, fy, c=farbe, label=Name, alpha = 0.7)
    # print(f'$Gauskurve {i+1}:', out.beta, out.sd_beta,  '$')
    print(f'{i+1}\t& {abs(out.beta[0]):.1f}$\pm${abs(out.sd_beta[0]):.1f} \t & {abs(Faktor*out.beta[1]):.1f}$\pm${abs(Faktor*out.sd_beta[1]):.1f} \t & {abs(out.beta[2]):.2f}$\pm${abs(Faktor*out.sd_beta[2]):.2f} \t & {abs(Faktor*out.beta[3]):.4f}$\pm${abs(Faktor*out.sd_beta[3]):.4f} \t & {abs(Faktor*out.beta[4]):.0f}$\pm${abs(out.sd_beta[4]):.0f} \t \\\\ ')
    # print('$R^2 =', r2_score(y, fy), '$')
    # print()
    return out.beta

ax, plt = plt.subplots()


# Gaus-Fits

Farbe = ['firebrick', 'orangered','gold','lightgreen','seagreen' ,'steelblue',  'darkorchid', 'orchid', 'palevioletred']
Beta = [[10000, 700, 15, 1, 1], 
        [10000, 2300, 20, 0, 40], 
        [10000, 3300, 20, 1, 30], 
        [3000, 5455, 10, 0, 30], 
        [5000, 5700, 10, 0, 30],
        [5000, 8525, 10, 0, 30],  
        [5000, 9045, 30, 0, 10], 
        [2000, 11595, 20, 0, 20],  
        [70000, 13700, 15, 1, 1]]
Fenster = [600, 800, 
           2100, 2500, 
           3100, 3400, 
           5420, 5530, 
           5660, 5730, 
           8450, 8600,
           8950, 9160,
           11550, 11650,
           13600, 13800]

# # Fenster / Params des Röntgenpeaks
# Beta = [[-25000, 3.5, 10, 0, 0]]
# Fenster=[0, 200]

for i in range(len(Beta)):
    TempxData = np.where((xData > Fenster[2*i+1]) | (xData < Fenster[2*i]), 0, xData)
    TempyData = np.where((xData > Fenster[2*i+1]) | (xData < Fenster[2*i]), 0, yData)
    out = gausfit(gaus, xData[Fenster[2*i]:Fenster[2*i+1]], yData[Fenster[2*i]:Fenster[2*i+1]], Farbe[i], Beta[i], f'Gauskurve {i+1}')

def Faktors(x):
    return Faktor*x

plt.scatter(xData, yData, s=2, c='black', label='Messwerte')
plt.legend()



# plt.set_ylim(0, max(yData)*1.05)

plt.set_xticks(np.linspace(0, 16000, 9))
plt.set_xticks(np.linspace(0, 16000, 33), minor=True)
plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.set_title('Kanalnummer', fontsize='large')
plt.set_ylabel('Anzahl Messungen pro Kanal', fontsize='large')
secax = plt.secondary_xaxis('bottom', functions=(Faktors, Faktors))
secax.set_xlabel('Energie / keV', fontsize='large')
secax.set_xticks(np.linspace(0, Faktor*16000, 9))
secax.set_xticks(np.linspace(0, Faktor*16000, 33), minor=True)
# plt.tick_params(axis='x', length=4, direction='in', width=4)
plt.axis()
plt.grid()
plt.set_xlim(0, 80)
ax.savefig(f'{Path}Boden Ohne.pdf')

print(Faktor*4)