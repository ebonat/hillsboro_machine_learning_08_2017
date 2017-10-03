
# coding: utf-8

# # LEAD QUAL Model

# Lead Qual Forecasting -- we wish to forecast incoming lead qual tickets to optimize staffing levels.

# In[1]:

import warnings
warnings.filterwarnings('ignore')
import pandas
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import keyring
import sqlalchemy
import cx_Oracle
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import arima_model as am


# In[2]:

filepath1 = './Train_data.csv'
filepath2 = './Test_data.csv'
def initialCSV(csv_filepath):
    dfRaw = pandas.read_csv(csv_filepath, index_col=0)
    dfRaw.index = pandas.to_datetime(dfRaw.index)
    dfRaw['Count_Tickets'] = dfRaw['Count_Tickets'].astype(float)
    return dfRaw
dfLeadsTrain = initialCSV(filepath1)
dfLeadsTest = initialCSV(filepath2)
dfLeadsTrain['Day_of_Week'] = dfLeadsTrain.index.dayofweek
dfLeadsTest['Day_of_Week'] = dfLeadsTest.index.dayofweek


# In[3]:

dfLeadsTrain.head(5).T


# Looking at the raw data:

# In[4]:

name = 'Day_of_Week'
dummies = pandas.get_dummies(dfLeadsTrain[name], prefix = name)
cols = dummies.columns
dfLeadsTrain[cols] = dummies
testdummies = pandas.get_dummies(dfLeadsTest[name], prefix = name)
testcols = testdummies.columns
dfLeadsTest[testcols] = testdummies


# In[5]:

dfLeadsTrain.Count_Tickets.plot()
plt.xticks(rotation=70)
plt.show()


# Let's also import some date information

# In[6]:


dfDates = pandas.read_csv('./Date_Table.csv')
dfDates.index = dfDates.CALENDAR_DATE[:]
dfDates = dfDates.drop('CALENDAR_DATE', axis = 1)
dfLeadsTrain = dfLeadsTrain.join(dfDates)
dfLeadsTest = dfLeadsTest.join(dfDates)
dfLeadsTrain.head().T


# Durbin Watson Statistic:

# In[7]:

sm.stats.durbin_watson(dfLeadsTrain.Count_Tickets)


# Plotting the Autocorrelation and Partial Autocorrelation functions:

# In[8]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dfLeadsTrain.Count_Tickets, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dfLeadsTrain.Count_Tickets, lags=40, ax=ax2)
plt.show()


# In[9]:

dfLeads2 = dfLeadsTrain[:]
col = (dfLeads2['Count_Tickets'] - dfLeads2['Count_Tickets'].mean()) / (dfLeads2['Count_Tickets'].std())
col = col.rename('Count_Tickets2')
dfLeads2 = pandas.concat([dfLeads2, col], axis =1)
plt.acorr(dfLeads2['Count_Tickets2'],maxlags = len(dfLeads2['Count_Tickets2']) -1, linestyle = "solid", usevlines = False, marker='')
plt.show()
autocorrelation_plot(dfLeadsTrain['Count_Tickets'])
plt.show()


# In[10]:

print dfLeadsTrain.columns
print dfLeadsTest.tail()


# In[11]:

exog_cols = [u'Day_of_Week_1',u'Day_of_Week_2', u'Day_of_Week_3', u'Day_of_Week_4', u'Day_of_Week_5',
       u'Day_of_Week_6',     u'MONTH_END', u'HOLIDAY']
modelARMA = am.ARMA(dfLeadsTrain.Count_Tickets, (2,1), exog=dfLeadsTrain[exog_cols])
resultsARMA = modelARMA.fit()
print (resultsARMA.summary())


# Durbin-Watson on results shows no/low positive autocorrelation

# In[12]:

print resultsARMA.aic, resultsARMA.bic, resultsARMA.hqic
print sm.stats.durbin_watson(resultsARMA.resid.values)


# In[13]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = resultsARMA.resid.plot(ax=ax)
plt.show()


# In[14]:

residARMA = resultsARMA.resid
stats.normaltest(residARMA)


# Residuals display odd discontinuity

# In[15]:

plt.plot(dfLeadsTrain['Count_Tickets'], residARMA, 'k*')
plt.show()


# In[16]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(residARMA, line='q', ax=ax, fit=True)
plt.show()


# ##### Ploting the ACF and PACF for the residuals, we see strong 5 day spikes we haven't accounted for.

# In[17]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(residARMA.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(residARMA, lags=40, ax=ax2)
plt.show()


# Next, we calculate the lag, autocorrelation (AC), Q statistic and Prob>Q. 

# In[18]:

r,q,p = sm.tsa.acf(residARMA.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pandas.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table)


# We note that the p-values for the Ljung–Box Q test all are well below .05 

# In[19]:

predict_leadsARMA = resultsARMA.predict(440,464, exog=dfLeadsTest[exog_cols], dynamic=True)
print predict_leadsARMA 


# In[20]:

ax = dfLeadsTrain['Count_Tickets'].ix[380:].plot(figsize=(12,8))
ax = dfLeadsTest['Count_Tickets'].plot(ax=ax, style='k--', label='Test')
ax = predict_leadsARMA .plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend()
plt.show()


# In[21]:

sm.tools.eval_measures.rmse(predict_leadsARMA[-22:],dfLeadsTest.Count_Tickets)


# In[22]:

fullpreds = resultsARMA.predict(1,len(dfLeadsTrain), exog=dfLeadsTrain[exog_cols], dynamic=False)
sm.tools.eval_measures.rmse(fullpreds, dfLeadsTrain.Count_Tickets)


# In[23]:

ax = dfLeadsTrain['Count_Tickets'].ix[380:].plot(figsize=(12,8))
ax = dfLeadsTest['Count_Tickets'].plot(ax=ax, style='k--', label='Test')
ax = fullpreds.ix[380:].plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend()
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



