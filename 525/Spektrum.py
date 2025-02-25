import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit 
import os
import scipy.odr as odr
import math
from sklearn.metrics import r2_score

path = "C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv" # PC
# path = 'C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\525\\Messungen\\csv' # Surface

ax, plt = plt.subplots()
def Plot(data, Name):
    plt.grid()
    plt.scatter(data['x'], data['y'], color ='cornflowerblue', marker='.', s=2)
    plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
    ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\{Name[:-4]}.pdf')
    ax.show
    plt.cla()


def gaus(Para, x): 
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2)



def gausfit(func, x, y, farbe, beta, Name):

    model = odr.Model(func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=1000)
    out = myodr.run()
    
    fy = func(out.beta, x)
    rsquared = r2_score(y, fy)
    # out.pprint()
    print('$Parameter:', out.beta, out.sd_beta,  '$')
    print('$R_{lin}^2 =',rsquared, '$')
    plt.plot(x, fy, c=farbe,label = Name )
    return out






# # Plot für alle

# for dateiname in os.listdir(path):
#     if os.path.isfile(os.path.join(path, dateiname)):
#         Data = pd.read_csv(f'{path}\\{dateiname}', sep="\t",header=0, names=['x', 'y'])
#         Plot(Data, dateiname)



# # Fünf Dinger in einem

# FünfNamen = ['Messung Na Promt 0ns', 'Messung Na Promt 16ns', 'Messung Na Promt 32ns', 'Messung Na Promt 48ns', 'Messung Na Promt 64ns']
# FünfFarben = ['sienna', 'darkgoldenrod', 'darkolivegreen', 'steelblue', 'orchid']
# LaufAdler = []
# for i in range(len(FünfNamen)):
#     fünf = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\{FünfNamen[i]}.csv', sep="\t",header=0, names=['x', 'y'])
#     plt.grid()
#     plt.scatter(fünf['x'], fünf['y'], color =FünfFarben[i], marker='.', s=2)
#     plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
#     beta = [9000, 800, 50]
#     gausfit(gaus, fünf['x'], fünf['y'], FünfFarben[i], beta, FünfNamen[i][17:])
#     ax.set_label(FünfNamen)
# plt.set_xlim(0, 1750)
# plt.legend()
# ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Na Prompt.pdf')
# plt.cla()



# # zwo Energiespektren Gaus

# FünfNamen = ['Messung links Na 2', 'Messung rechts Na 2']
# FünfFarben = ['darkslategrey', 'darkslategrey', 'firebrick', 'firebrick']
# LaufAdler = []
# for i in range(len(FünfNamen)):
#     fünf = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\{FünfNamen[i]}.csv', sep="\t",header=0, names=['x', 'y'])
#     plt.grid()
#     plt.scatter(fünf['x'], fünf['y'], color =FünfFarben[i], marker='.', s=2)
#     plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
#     beta = [50000, 7000, 50]
#     beta1 = [5000, 3000, 250]
#     out = gausfit(gaus, fünf['x'][6000:], fünf['y'][6000:], FünfFarben[i+2], beta)
#     out1 = gausfit(gaus, fünf['x'][:6000], fünf['y'][:6000], FünfFarben[i+2], beta1)
#     plt.plot(fünf['x'], (gaus(out.beta, fünf['x'])+gaus(out1.beta, fünf['x'])), c=FünfFarben[i+2], label='Lineare Anpassung')
#     ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\{FünfNamen[i]}Fit.pdf')
#     plt.cla()




# # 511, 81 und 356 linien fit

# FünfNamen = ['Messung Ba 81 rechts', 'Messung Ba 356 links', 'Messung rechts Na 511', 'Messung links Na 511']
# FünfFarben = ['orchid', 'navy']
# LaufAdler = []
# for i in range(len(FünfNamen)):
#     fünf = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\{FünfNamen[i]}.csv', sep="\t",header=0, names=['x', 'y'])
#     plt.grid()
#     plt.scatter(fünf['x'], fünf['y'], color =FünfFarben[0], marker='.', s=2)
#     plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
#     keke = [1200, 4500, 7500, 7500]
#     beta = [50000, keke[i] , 50]
#     out = gausfit(gaus, fünf['x'], fünf['y'], FünfFarben[1], beta)
#     plt.plot(fünf['x'], gaus(out.beta, fünf['x']), c=FünfFarben[1], label='Lineare Anpassung')
#     ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\{FünfNamen[i]} Fit.pdf')
#     plt.cla()




# # SCA und CFD Vergleich

# FünfNamen = ['Messung Ba fast links', 'Messung Ba fast links 2', 'Messung Ba fast rechts', 'Messung Ba fast rechts 2']
# FünfFarben = ['steelblue', 'goldenrod']
# LaufAdler = []
# for i in range(2):
#     fünf = pd.read_csv(f'C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\525\\Messungen\\csv\\{FünfNamen[2*i]}.csv', sep="\t",header=0, names=['x', 'y'])
#     sechs = pd.read_csv(f'C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\525\\Messungen\\csv\\{FünfNamen[2*i+1]}.csv', sep="\t",header=0, names=['x', 'y'])
#     plt.grid()
#     plt.scatter(fünf['x'], fünf['y'], color =FünfFarben[0], marker='.', s=2, label='Ohne CFD Wert' )
#     plt.scatter(sechs['x'], sechs['y'], color =FünfFarben[1], marker='.', s=2, label='Mit CFD Wert')
#     plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
#     plt.legend()
#     ax.savefig(f'C:\\Users\\Surface Pro 7 Manni\\Desktop\\Code Dateien\\P5\\525\\pdf\\{FünfNamen[2*i][:-1]} Fit.pdf')
#     plt.cla()



# # CFD Schwelle

# Namen = ['Messung Ba fast links 2', 'Messung Ba fast rechts 2']
# Farbe = ['steelblue', 'goldenrod']

# def Erff(Para, x):
#     temp = []
#     for i in x:
#         temp2 = Para[0] * (special.erf((i - Para[1])/Para[2])+1)
#         temp.append(temp2)
#     return np.array(temp)
     
# for i in range(2):
#     data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\{Namen[i]}.csv', sep="\t",header=0, names=['x', 'y'])
#     plt.grid()
#     plt.scatter(data['x'], data['y'], color =Farbe[0], marker='.', s=2, label='Werte')
#     Grenze = [800, 1300]
#     Oben = [250, 180]
#     xWerte = np.linspace(0, Grenze[i], 100)
#     beta = [70., 400., 1.]
#     yWerte = Erff(gausfit(Erff, data['x'][:Grenze[i]], data['y'][:Grenze[i]], Farbe[1], beta).beta, xWerte)
#     plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
#     plt.legend()
#     plt.set_xlim(-50, Grenze[i]+50)
#     plt.set_ylim(-20, Oben[i])
#     ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\{Namen[i]} Fit nah.pdf')
#     plt.cla()







def Rfunc(Para, x):
    return Para[0]*np.exp(-Para[1]*x)


data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\Messung Lebensdauer.csv', sep="\t",header=0, names=['x', 'y'])

plt.grid()
beta = [120, 1e-3]
out = gausfit(Rfunc, data['x'][1500:3000], data['y'][1500:3000], 'navy', beta, 'Lebenszeitkurve')
plt.scatter(data['x'][1000:3000], data['y'][1000:3000], color='darkolivegreen', marker='.', s=2, label='Messwerte')
plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
plt.legend()
ax.savefig('C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Lebensdauer Fit 1500-3000.pdf')
plt.cla()


data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\Messung Lebensdauer.csv', sep="\t",header=0, names=['x', 'y'])
plt.grid()
beta = [120, 1e-3]
out = gausfit(Rfunc, data['x'][1650:3000], data['y'][1650:3000], 'navy', beta, 'Lebenszeitkurve')
plt.scatter(data['x'][1000:3000], data['y'][1000:3000], color='darkolivegreen', marker='.', s=2, label='Messwerte')
plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
plt.legend()
ax.savefig('C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Lebensdauer Fit 1650-3000.pdf')
plt.cla()


data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\Messung Lebensdauer.csv', sep="\t",header=0, names=['x', 'y'])
plt.grid()
beta = [120, 1e-3]
out = gausfit(Rfunc, data['x'][1800:3000], data['y'][1800:3000], 'navy', beta, 'Lebenszeitkurve')
plt.scatter(data['x'][1000:3000], data['y'][1000:3000], color='darkolivegreen', marker='.', s=2, label='Messwerte')
plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
plt.legend()
ax.savefig('C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Lebensdauer Fit 1800-3000.pdf')
plt.cla()


data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\Messung Lebensdauer.csv', sep="\t",header=0, names=['x', 'y'])
plt.grid()
beta = [120, 1e-3]
out = gausfit(Rfunc, data['x'][1950:3000], data['y'][1950:3000], 'navy', beta, 'Lebenszeitkurve')
plt.scatter(data['x'][1000:3000], data['y'][1000:3000], color='darkolivegreen', marker='.', s=2, label='Messwerte')
plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
plt.legend()
ax.savefig('C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Lebensdauer Fit 1950-3000.pdf')
plt.cla()


data = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\Messung Lebensdauer.csv', sep="\t",header=0, names=['x', 'y'])
plt.grid()
beta = [120, 1e-3]
out = gausfit(Rfunc, data['x'][2100:3000], data['y'][2100:3000], 'navy', beta, 'Lebenszeitkurve')
plt.scatter(data['x'][1000:3000], data['y'][1000:3000], color='darkolivegreen', marker='.', s=2, label='Messwerte')
plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
plt.legend()
ax.savefig('C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Lebensdauer Fit 2100-3000.pdf')
plt.cla()

# print(np.log(2)/(16.4*(out.beta[1])))
# print(np.log(2)*np.sqrt((16.4 * out.sd_beta[1])**2 + (0.6*out.beta[1])**2)/(16.4*(out.beta[1]))**2)