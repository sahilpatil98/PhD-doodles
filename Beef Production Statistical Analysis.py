# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 07:44:44 2020

@author: SahilPatil
"""

# Part (a) Discrete State Growth Model
import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt 

# preference and production parameters
beta = 0.96                          # Discount Factor
A = 1.00                             # TFP
alpha = 0.36                         # Capital share
delta = 1.00                         # depreciation rate

# capital grid for the current period
kss = ((1.0/beta - 1.0 + delta)/(alpha*A))**(1.0/(alpha - 1.0)) # steady state capital stock
kMin = 0.8*kss                      # lower bound of capital
kMax = 1.2*kss                      # upper bound of capital 
nk = 100                            # number of gridpoints
kgrid = np.linspace(kMin,kMax,nk).reshape(nk,1)       # grid for capital

# capital grid for the next period
kprimeGrid = np.copy(kgrid)         # grid for kprime

# set up convergence criterion
iterMax = 1000                     # max number of iterations
tol = 1e-6                         # tolerance 

# initialization of value and policy function
v = np.zeros((nk,1))              # v(i): value function at kgrid(i)
dec = np.zeros((nk,1))            # dec(i): kprime given current kgrid(i)

# current-period flow utility
totalResource = A*kgrid**alpha + (1.0 - delta)*kgrid # totalResource(i): total available resource at kgrid(i)
consMat = totalResource - kprimeGrid.T  # consMat(i,j): consumption matrix for kgrid(i) and kprimeGrid(j)
isFeasible = (consMat > 0.0)         # isFeasible(i,j): indicator of feasible consumption for kgrid(i) and kprimeGrid(j)
util = np.tile(-np.inf,(nk,nk));  # util(i,j): pre-allocating util(i,j) as -inf for kgrid(i) and kprimeGrid(j)
util[isFeasible] = np.log(consMat[isFeasible])   # util(i,j): current utility at kgrid(i) and kprimeGrid(j)

# iterate over the value function
print('######################################')
print('Starting Value Function Iteration:')
print('######################################\n')
print('Iter','Metric')

for iter in range(iterMax):
    # maximize Bellman Equation
    bellmanMat = util + beta*v.T            # bellmanMat(i,j): the value in Bellman Equation for kgrid(i) and kprimeGrid(j)
    tv = np.max(bellmanMat,axis = 1)        # tv(i): the max Bellman value for kgrid(i)
    ind = np.argmax(bellmanMat,axis = 1)    # ind(i): the optimal kprime index for kgrid(i)
    tdec = kgrid[ind]                       # tdec(i): the optimal kprime
    
    # calculate convergence metric and update
    metric = np.amax(np.abs(tv-v))          # metric for convergence
    print(iter,metric)                      # print iteration metric 
    
    if (metric <= tol): # Converged: Stop
        break
    else: # Not Converged: Continue
        v = np.copy(tv) # Update value function
        dec = np.copy(tdec) # Update policy function


##PART (b)
# value function in closed-form solution: v(k) = E + F log(k)
E = (np.log(A*(1-alpha*beta)) + (alpha*beta/(1-alpha*beta))*np.log(A*alpha*beta))/(1-beta)
F = alpha/(1-alpha*beta)
vClosedForm = E + F*np.log(kgrid) # closed-form solution

# policy function in closed-form solution: kprime(k) = alpha*beta*A*k**alpha
decClosedForm = alpha*beta*A*(kgrid**alpha)
        
##PART (c)
# Plot value function
fig1 = plt.figure()                      # generate figure object
ax1 = fig1.add_subplot(1,1,1)            # generate plot object
ax1.plot(kgrid,v)                        # plot value function
ax1.plot(kgrid,vClosedForm)              # plot closed-form solution
ax1.set_title('Value Function')          # the title of the plot
ax1.set_xlabel('Capital')                # the label for horizontal axis
ax1.set_ylabel('Value')                  # the label for vertical axis
ax1.legend(labels = ['v','Closed Form']) # the legend for two plots
plt.show()                               # display the plot

fig2 = plt.figure()                      # generate figure object
ax2 = fig2.add_subplot(1,1,1)            # generate plot object
ax2.plot(kgrid,dec)                      # plot value function
ax2.plot(kgrid,decClosedForm)            # plot closed-form solution
ax2.set_title('Policy Function')         # the title of the plot
ax2.set_xlabel('Capital')                # the label for horizontal axis
ax2.set_ylabel('Capital Next Period')    # the label for vertical axis
ax2.legend(labels = ['kprime','Closed Form']) # the legend for two plots
plt.show()                                    # display the plot


##Part (d)

# capital grid for the current period
kss = ((1.0/beta - 1.0 + delta)/(alpha*A))**(1.0/(alpha - 1.0)) # steady state capital stock
kMin = 0.8*kss                      # lower bound of capital
kMax = 1.2*kss                      # upper bound of capital 
nk = 500                            # number of gridpoints
kgrid = np.linspace(kMin,kMax,nk).reshape(nk,1)       # grid for capital

# capital grid for the next period
kprimeGrid = np.copy(kgrid)         # grid for kprime

# set up convergence criterion
iterMax = 1000                     # max number of iterations
tol = 1e-6                         # tolerance 

# initialization of value and policy function
v = np.zeros((nk,1))              # v(i): value function at kgrid(i)
dec = np.zeros((nk,1))            # dec(i): kprime given current kgrid(i)

# current-period flow utility
totalResource = A*kgrid**alpha + (1.0 - delta)*kgrid # totalResource(i): total available resource at kgrid(i)
consMat = totalResource - kprimeGrid.T  # consMat(i,j): consumption matrix for kgrid(i) and kprimeGrid(j)
isFeasible = (consMat > 0.0)         # isFeasible(i,j): indicator of feasible consumption for kgrid(i) and kprimeGrid(j)
util = np.tile(-np.inf,(nk,nk));  # util(i,j): pre-allocating util(i,j) as -inf for kgrid(i) and kprimeGrid(j)
util[isFeasible] = np.log(consMat[isFeasible])   # util(i,j): current utility at kgrid(i) and kprimeGrid(j)

# iterate over the value function
print('######################################')
print('Starting Value Function Iteration:')
print('######################################\n')
print('Iter','Metric')

for iter in range(iterMax):
    # maximize Bellman Equation
    bellmanMat = util + beta*v.T            # bellmanMat(i,j): the value in Bellman Equation for kgrid(i) and kprimeGrid(j)
    tv = np.max(bellmanMat,axis = 1)        # tv(i): the max Bellman value for kgrid(i)
    ind = np.argmax(bellmanMat,axis = 1)    # ind(i): the optimal kprime index for kgrid(i)
    tdec = kgrid[ind]                       # tdec(i): the optimal kprime
    
    # calculate convergence metric and update
    metric = np.amax(np.abs(tv-v))          # metric for convergence
    print(iter,metric)                      # print iteration metric 
    
    if (metric <= tol): # Converged: Stop
        break
    else: # Not Converged: Continue
        v = np.copy(tv) # Update value function
        dec = np.copy(tdec) # Update policy function

# value function in closed-form solution: v(k) = E + F log(k)
E = (np.log(A*(1-alpha*beta)) + (alpha*beta/(1-alpha*beta))*np.log(A*alpha*beta))/(1-beta)
F = alpha/(1-alpha*beta)
vClosedForm = E + F*np.log(kgrid) # closed-form solution

# policy function in closed-form solution: kprime(k) = alpha*beta*A*k**alpha
decClosedForm = alpha*beta*A*(kgrid**alpha)
        
# Plot value function
fig1 = plt.figure()                      # generate figure object
ax1 = fig1.add_subplot(1,1,1)            # generate plot object
ax1.plot(kgrid,v)                        # plot value function
ax1.plot(kgrid,vClosedForm)              # plot closed-form solution
ax1.set_title('Value Function')          # the title of the plot
ax1.set_xlabel('Capital')                # the label for horizontal axis
ax1.set_ylabel('Value')                  # the label for vertical axis
ax1.legend(labels = ['v','Closed Form']) # the legend for two plots
plt.show()                               # display the plot

fig2 = plt.figure()                      # generate figure object
ax2 = fig2.add_subplot(1,1,1)            # generate plot object
ax2.plot(kgrid,dec)                      # plot value function
ax2.plot(kgrid,decClosedForm)            # plot closed-form solution
ax2.set_title('Policy Function')         # the title of the plot
ax2.set_xlabel('Capital')                # the label for horizontal axis
ax2.set_ylabel('Capital Next Period')    # the label for vertical axis
ax2.legend(labels = ['kprime','Closed Form']) # the legend for two plots
plt.show()                                    # display the plot

#You can see that the kprime is almost too small to see on the graph of k_t+1 and k_t and that you cannot see it on the graph.