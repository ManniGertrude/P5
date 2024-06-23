import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit 
import os
import scipy.odr as odr
from sklearn.metrics import r2_score

path = "C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv"

ax, plt = plt.subplots()
def Plot(data, Name):
    plt.grid()
    plt.scatter(data['x'], data['y'], color ='cornflowerblue', marker='.')
    plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
    ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\{Name[:-4]}.pdf')
    ax.show
    plt.cla()


def gaus(Para, x): 
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2)



def gausfit(func, x, y, farbe):

    model = odr.Model(func)
    mydata = odr.RealData(x, y)
    myodr = odr.ODR(mydata, model, beta0=[9000, 800, 50], maxit=1000, )
    out = myodr.run()
    fy = gaus(out.beta, x)
    rsquared = r2_score(y, fy)
    # out.pprint()
    # print('$Parameter:', out.beta, out.sd_beta,  '$')
    print('$R_{lin}^2 =',rsquared, '$')
    plt.plot(x, fy, c=farbe, label='Lineare Anpassung')









# for dateiname in os.listdir(path):
#     if os.path.isfile(os.path.join(path, dateiname)):
#         Data = pd.read_csv(f'{path}\\{dateiname}', sep="\t",header=0, names=['x', 'y'])
#         Plot(Data, dateiname)

FünfNamen = ['Messung Na Promt 0ns', 'Messung Na Promt 16ns', 'Messung Na Promt 32ns', 'Messung Na Promt 48ns', 'Messung Na Promt 64ns']
FünfFarben = ['darkslategrey', 'mediumorchid', 'firebrick', 'burlywood', 'cornflowerblue']
LaufAdler = []
for i in range(len(FünfNamen)):
    fünf = pd.read_csv(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\Messungen\\csv\\{FünfNamen[i]}.csv', sep="\t",header=0, names=['x', 'y'])
    plt.grid()
    plt.scatter(fünf['x'], fünf['y'], color =FünfFarben[i], marker='.', s=2)
    plt.set(xlabel='Kanalnummer', ylabel='Anzahl an Messergebnissen')
    gausfit(gaus, fünf['x'], fünf['y'], FünfFarben[i])
plt.set_xlim(0, 1750)
ax.savefig(f'C:\\Users\\kontr\\Desktop\\Github\\P5\\525\\pdf\\Na Prompt.pdf')
ax.show













