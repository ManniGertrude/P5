import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit 

# LBa1 = pd.read_csv("P5\\525\\Messungen\\Messung links Ba 1.csv", sep="\t",header=0, names=['x', 'y'])
# LBa1Name = ['Ba 1 links']

# RBa1 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Ba 1.csv", sep="\t",header=0, names=['x', 'y'])
# RBa1Name = ['Ba 1 rechts']

# # LNa1 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 1.csv", sep="\t",header=0, names=['x', '0', 'y'])
# # LNa1Name = ['Na 1 links']

# LNa2 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 2.csv", sep="\t",header=0, names=['x', 'y'])
# LNa2Name = ['Na 2 links']

# LNa511 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 511.csv", sep="\t",header=0, names=['x', 'y'])
# LNa511Name = ['Na 511 links']

# RNa1 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 1.csv", sep="\t",header=0, names=['x', 'y'])
# RNa1Name = ['Na 1 rechts']

# RNa2 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 2.csv", sep="\t",header=0, names=['x', 'y'])
# RNa2Name = ['Na 2 rechts']

# RNa511 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 511.csv", sep="\t",header=0, names=['x', 'y'])
# RNa511Name = ['Na 511 rechts']

LfastBa= pd.read_csv("P5\\525\\Messungen\\Messung Ba fast links.csv", sep="\t",header=0, names=['x', 'y'])
LfastBaName = ['Na 511 rechts']

RfastBa = pd.read_csv("P5\\525\\Messungen\\Messung Ba fast rechts.csv", sep="\t",header=0, names=['x', 'y'])
RfastBaName = ['Na 511 rechts']




def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))









ax, plt = plt.subplots()
def Plot(data, Name):
    # parameters, covariance = curve_fit(gauss, data['x'], data['y']) 
    # fit_y = gauss(data['x'], parameters[0], parameters[1], parameters[2], parameters[3], bounds=([ , , , ], [ , , , ])) 
    plt.grid()
    # plt.plot(data['x'], fit_y)
    plt.scatter(data['x'], data['y'], color ='cornflowerblue', marker='+', label = Name[0])
    plt.set(xlabel='x', ylabel='y')
    plt.legend()
    ax.savefig(f'P5\\525\\{Name[0]}.pdf')
    ax.show
    plt.cla()


Plot(LfastBa, LfastBaName)
Plot(RfastBa, RfastBaName)

