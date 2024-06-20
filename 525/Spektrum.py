import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd

LBa1 = pd.read_csv("P5\\525\\Messungen\\Messung links Ba 1.csv", sep="\t",header=0, names=['x', 'y'])
LBa1Name = ['Ba 1 links']

RBa1 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Ba 1.csv", sep="\t",header=0, names=['x', 'y'])
RBa1Name = ['Ba 1 rechts']

# LNa1 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 1.csv", sep="\t",header=0, names=['x', '0', 'y'])
# LNa1Name = ['Na 1 links']

LNa2 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 2.csv", sep="\t",header=0, names=['x', 'y'])
LNa2Name = ['Na 2 links']

LNa511 = pd.read_csv("P5\\525\\Messungen\\Messung links Na 511.csv", sep="\t",header=0, names=['x', 'y'])
LNa511Name = ['Na 511 links']

RNa1 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 1.csv", sep="\t",header=0, names=['x', 'y'])
RNa1Name = ['Na 1 rechts']

RNa2 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 2.csv", sep="\t",header=0, names=['x', 'y'])
RNa2Name = ['Na 2 rechts']

RNa511 = pd.read_csv("P5\\525\\Messungen\\Messung rechts Na 511.csv", sep="\t",header=0, names=['x', 'y'])
RNa511Name = ['Na 511 rechts']



ax, plt = plt.subplots()
def Plot(data, Name):
    plt.grid()
    plt.scatter(data['x'], data['y'], color ='cornflowerblue', marker='+', label = Name[0])
    plt.set(xlabel='x', ylabel='y')
    plt.legend()
    ax.savefig(f'P5\\525\\{Name[0]}.pdf')
    ax.show
    plt.cla()




Plot(LBa1, LBa1Name)
Plot(RBa1, RBa1Name)
# Plot(LNa1, LNa1Name)
Plot(LNa2, LNa2Name)
Plot(RNa1, RNa1Name)
Plot(RNa2, RNa2Name)
Plot(LNa511, LNa511Name)
Plot(RNa511, RNa511Name)

