# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:10:32 2021

@author: d0tamon
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats
from scipy.stats import t

colnames = ['year','q_hh','p_hh','ir','trm','tc','mhc','cpi','shp']
data = pd.read_csv(r"C:\Users\d0tamon\Desktop\housing.txt", delimiter = "	", header = None, names = colnames)
model = pd.DataFrame()
model['ln_q_hh'] = np.log(data['q_hh'])
model['ln_p/cpi'] = np.log(data['p_hh']/data['cpi'])
model['ln_ir'] = np.log(data['ir'])
model['ln_mhc/tc'] = np.log(data['mhc']/data['tc'])
model['ln_shp'] = np.log(data['shp'])


#Variable description
X, y = model[['ln_p/cpi','ln_ir','ln_mhc/tc','ln_shp']], model[['ln_q_hh']]
X = sm.add_constant(X)

#(a) Descriptive Statistics
descriptive_stats = data.describe()

#(b) Model
ols = sm.OLS(y, X)
ols_result = ols.fit()
ols_result.summary(alpha = 0.05)

coefficients = ols_result.params
standard_errors = ols_result.bse

#Interpretation
# If we change real prices by 1%, given interest rate, ratio of manufactured housing credit by total installment credit, and shipment of new manufactured homes are held constant, the total quantity demanded of new manufactured homes for residential use decreases by 0.0254%
# If we change interest rate by 1%, given real prices, ratio of manufactured housing credit by total installment credit, and shipment of new manufactured homes are held constant, the total quantity demanded of new manufactured homes for residential use decreases by 0.0549%
# If we change ratio of manufactured housing credit by total installment credit by 1%, given real prices, interest rate, and shipment of new manufactured homes are held constant, the total quantity demanded of new manufactured homes for residential use increases by 0.0591%
# If we change shipment of new manufactured homes by 1%, given real prices, interest rate, and ratio of manufactured housing credit by total installment credit are held constant, the total quantity demanded of new manufactured homes for residential use increases by 0.8353%


#(c) 
#Null and Alternate Hypothesis

#H_0 = beta_1 = 0 vs H_a = beta_1 != 0 -> Beta 1 = coefficient on constant
#H_0 = beta_2 = 0 vs H_a = beta_2 != 0 -> Beta 2 = coefficient on ln_p/cpi
#H_0 = beta_3 = 0 vs H_a = beta_3 != 0 -> Beta 3 = coefficient on ln_ir
#H_0 = beta_4 = 0 vs H_a = beta_4 != 0 -> Beta 4 = coefficient on ln_mhc/tc
#H_0 = beta_5 = 0 vs H_a = beta_4 != 5 -> Beta 5 = coefficient on ln_shp


#Critical  for alpha = 0.05
p_high = 0.975
p_low = 0.025
df = 19
t_crit_low = t.ppf(p_low,df)
t_crit_high = t.ppf(p_high,df)
t_crit = [t_crit_low,t_crit_high]

#Test Results
t_values = ols_result.tvalues

#Do not reject beta 1
#Do not reject beta 2
#Do not reject beta 3
#Do not reject beta 4
#Reject beta 5


#(d)
#H_0 = beta_2 = 0, beta_3 = 0, beta_4 = 0, beta_5 = 0 vs H_a = Either one of the betas is not equal to zero

f_crit = scipy.stats.f.ppf(q = 1-0.05, dfn = 4, dfd = 16)

r_mat = np.identity(len(ols_result.params))
r_mat = r_mat[1:,:]
f_test = ols_result.f_test(r_mat)
f_value = f_test.fvalue

#Reject H_0 showing at atleast one of the 4 betas is significant to the model that we are trying to predict.


#(e)
p_elasticity = coefficients[1]
glennon_p_elasticity = -0.58

#Glennon has a significantly high price elasticity compared to the elasticity we found in the model which was -0.02536. Glennon believes that prices are a significantly elastic when the data shows that they are not

#(f)
#Limitations may include overestimating the model since the r-squared in this case is 0.961.
#Too many variables may cause an issue while predicting the demand for houses for residential use
#Location and availability of that location to other crucial services may also be driving factors of new manufactured homes placed for residential use which is not covered in the model.