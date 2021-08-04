# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:13:06 2021

@author: d0tamon
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_excel(r"C:\Users\d0tamon\Downloads\HW2 2021.xlsx")

#(a) Descriptive statistics
descriptive_statistics = data.describe()

#(b) Model Estimation
X, y = data[['p_maize','Expenditure']], data[['q_maize']]
X = sm.add_constant(X)



ols = sm.OLS(y, X)
ols_result = ols.fit()
ols_result.summary()
### Coefficient Estimates, Standard Errors and t-statistics
coefficients = ols_result.params
standard_errors = ols_result.bse
t_stats = ols_result.tvalues


### Interpretation
## All the beta coefficients have p-value of less than 0.05 and therefore are significant. This means price and income both have a significant impact on the quantity of maize (as expected). 
##Adjusted R^2 -> The regression model accounts for 46.5% variation in the quantity of maize given 2 variables were used in the regression, namely prize of maize and total income.




#(c)
## const -> Given income and prize of the maize is 0, the expected value of quantity of maize is 4.3579.
## p_maize -> Given income is held constant, one unit increase in the price of maize decreases the quantity of maize by 0.0265 units. The parameter is statistically significant since the t-statistics is -3.22
## expenditure -> Give the prize of maize is held constant, one unit increase in income increases the quatity of maize by 0.0019. The parameter is statistically significant since the t-statistics is 9.657



#(d)
y_hat = pd.DataFrame(ols_result.predict(X), columns = ['y_hat'])


elasticity = pd.merge(data,y_hat, left_index=True,right_index=True)

inc_elast = []
price_elast = []

for i,j in zip(elasticity['q_maize'],elasticity['p_maize']):
    price_elast_calc = coefficients[1]*(j/i)
    price_elast.append(price_elast_calc)

for k,l in zip(elasticity['q_maize'],elasticity['Expenditure']):
    inc_elast_calc = (coefficients[2]*l)/k
    inc_elast.append(inc_elast_calc)
    
plt.hist(inc_elast)
plt.hist(price_elast)

#
min_inc_elast = min(inc_elast)
max_inc_elast = max(inc_elast)
mean_inc_elast = np.mean(inc_elast)
min_price_elast = min(price_elast)
max_price_elast = max(price_elast)
mean_price_elast = np.mean(price_elast)


data['income_elasticity'] = inc_elast
data['price_elasticity'] = price_elast

#Interpretation of average income and price elasticities
#On an average, an increase in the price of the maize by 1 unit, causes a decrease in quantity by 0.622 units of maize
#On an average, an increase in the expenditure on maize by 1 unit, causes an increase in quantity demanded by by 1.129 units of maize

inc_elast_at_mean = coefficients[2]*(np.mean(data['Expenditure'])/np.mean(data['q_maize']))
price_elast_at_mean = coefficients[1]*(np.mean(data['p_maize'])/np.mean(data['q_maize']))

#Interpretation of =income and price elasticities at the mean
#An increase of 1 unit of income, causes an increase of the quantity of maize by 0.8644 units given the quantity is 8.4895 and income is 3918.019
#An increase of 1 unit of prize, causes a decrease of the quantity of maize by -0.3777 units of maize given the quantity is 8.4895 and price of maize is 120.88

##The price elasticity at the mean is comparatively lower than the average price elasticity whereas the opposite is true for income elasticities, as income elasticity at the mean is higher than the average income elasticity. 
##The price elasticity at the mean is -0.377 whereas the average price elasticity is -0.6218.
##The income elasticity at the mean is 0.8644 whereas the average income elasticity is 0.33184.