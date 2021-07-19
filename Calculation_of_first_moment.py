import numpy as np
import sympy as sp
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import math

df = pd.read_csv('try.dat', delim_whitespace=True, header=None, names= ["z", "pL"])
# select first column and make list
z = df ['z'].tolist()
# select second column and make list
pL = df ['pL'].tolist()
def Average(pL): 
return sum(pL) / len(pL)
pN = Average(pL)
print("Average of the list =", round(pN, 2))
# define function v using lambda
v = lambda pL : pL
from scipy import integrate
#pN = (np.trapz(v(pL), z))
df['pN-pL'] = pN - df['pL']
df['multi'] = df['pN-pL'] * df['z']
plt.plot(z, df['pN-pL'])

#z1 = df['z']
def Reverse(lst):
new_lst = lst[::-1]
return new_lst
A = df['z'][0:48].tolist()
#print(Reverse(A))
z11 = Reverse(A)
#z11 = df['z'][0:48].tolist()
#z11.reverse()
z12 = df['z'][47:].tolist()
# read two halfes from multi pz
B = df['multi'] [0:48].tolist()
pz1 = Reverse(B)
pz2 = df['multi'] [47:].tolist()
# define function w1 using lambda
w1 = lambda pz1 : pz1
from scipy import integrate
integ1 = np.trapz(w1(pz1), z11)
# define function w2 using lambda
w2 = lambda pz2 : pz2
from scipy import integrate
integ2 = np.trapz(w2(pz2), z12)
result = integ1+integ2
print("final integration result two split:" '{}'.format (result))

