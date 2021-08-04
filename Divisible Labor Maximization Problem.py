# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:40:57 2021

@author: sahilpatil
"""

import numpy as np

#Coeffs
A = 1
alpha = 0.36
delta = 1
beta = 0.96

#
kss = ((1.0/beta - 1.0 + delta)/(alpha*A))**(1.0/(alpha - 1.0))
k_min = 0.8*kss
k_max = 1.2*kss
nk = 2
kgrid = np.linspace(k_min,k_max,nk).reshape(nk,1)

#
k_p_grid = np.copy(kgrid)

#
iterations = 1000
tol = 2e-5

#
v = np.zeros((nk,1))
dec = np.zeros((nk,1))

#
total_resource = A*kgrid**alpha + (1 - delta)*kgrid
consMat = total_resource - k_p_grid.T
isfeasible = (consMat > 0.0)
util = np.tile(-np.inf,(nk,nk))
util[isfeasible] = np.log(consMat[isfeasible])


#
for iter in range(iterations):
    bellman = util + beta*v.T
    tv = np.max(bellman, axis = 1)
    ind = np.argmax(bellman, axis = 1)
    tdec = kgrid[ind]
    
    metric = np.amax(np.abs(tv-v))
    print(iter,metric)
    
    if (metric < tol):
        break
    else:
        v = np.copy(tv)
        dec = np.copy(tdec)
        
##PART (b)
# value function in closed-form solution: v(k) = E + F log(k)
E = (np.log(A*(1-alpha*beta)) + (alpha*beta/(1-alpha*beta))*np.log(A*alpha*beta))/(1-beta)
F = alpha/(1-alpha*beta)
vClosedForm = E + F*np.log(kgrid) # closed-form solution

# policy function in closed-form solution: kprime(k) = alpha*beta*A*k**alpha
decClosedForm = alpha*beta*A*(kgrid**alpha)