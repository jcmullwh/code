# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:15:19 2021

@author: Game
"""
import json
import requests
import pandas as pd
import urllib.request
import seaborn as sns
import numpy as np


from IPython import get_ipython
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from functools import reduce
import win32api
import scipy.stats as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%% Functions

# Find lead/lag returns from a dict of values

def lead_lag_return(price_data, return_dict, price_data_column,date_column):

    lead_lag_return_data = pd.DataFrame()
    lead_lag_return_data['Same-Day'] = original_data[price_data_column]

    for i,j in enumerate(return_dict.keys()):
        column_name = j + " Return"
        lead_lag_return_data[column_name] = 0
        for k,l in enumerate(lead_lag_return_data['Same-Day']):
            if return_dict[j] > 0:
                try:
                    lead_lag_return_data.iloc[k,i+1]= (lead_lag_return_data.iloc[k+return_dict[j],0]-lead_lag_return_data.iloc[k,0])/lead_lag_return_data.iloc[k,0]
                except:
                   pass
            if return_dict[j] < 0 and (k + return_dict[j]) >= 0:
                try:
                    lead_lag_return_data.iloc[k,i+1]= (lead_lag_return_data.iloc[k,0]-lead_lag_return_data.iloc[k+return_dict[j],0])/lead_lag_return_data.iloc[k+return_dict[j],0]
                except:
                    pass

    lead_lag_return_data['DATE'] = original_data[date_column]
    
    return lead_lag_return_data

#Find lead/lag min price from dict of values

def lead_lag_min_price(original_data, period_dict, price_column, date_column):
    price_min = pd.DataFrame()
    price_min['Same-Day'] = original_data[price_column]
    for i,j in enumerate(period_dict.keys()):
        if period_dict[j] > 0:
            price_min_column_name = j + " Price Min"
            price_min[price_min_column_name] = price_min['Same-Day'].rolling(window=period_dict[j]).min().shift(period_dict[j]*-1)
        if period_dict[j] < 0:
            price_min_column_name = j + " Price Min"
            price_min[price_min_column_name] = price_min['Same-Day'].rolling(window=period_dict[j]*-1).min()
    price_min['DATE']=original_data[date_column]

    return price_min

# find lead/lag max price from a dict of values

def lead_lag_max_price(original_data, period_dict, price_column, date_column):
    price_max = pd.DataFrame()
    price_max['Same-Day'] = original_data[price_column]
    for i,j in enumerate(period_dict.keys()):
        if period_dict[j] > 0:
            column_name = j + " Price Min"
            price_max[column_name] = price_max['Same-Day'].rolling(window=period_dict[j]).max().shift(period_dict[j]*-1)
        if period_dict[j] < 0:
            column_name = j + " Price Min"
            price_max[column_name] = price_max['Same-Day'].rolling(window=period_dict[j]*-1).max()
    price_max['DATE']=original_data[date_column]

    return price_max

# find by what percentage the metrics are deviating from periods' average

def average_deviation(metrics,periods_dict, label):

    new_metrics = {}
    
    for p in periods_dict.keys():

        avg_metrics = pd.DataFrame(metrics['DATE'])
        avg_metrics_deviation = pd.DataFrame(metrics['DATE'])
        

        metrics = metrics.replace([np.inf, -np.inf, np.nan], 0)

        for c in metrics.columns:
            try:
                avg_metrics[c] = metrics.loc[:,c].rolling(window=periods_dict[p]).mean()
            except:
                pass
            
            avg_metrics = avg_metrics.replace([np.inf, -np.inf, np.nan], 0)
            

            for c in metrics.columns:
                try:
                    name = "Deviation from "+ p+ " " + label + " " + c
                    avg_metrics_deviation[name] = metrics.loc[:,c].sub(avg_metrics.loc[:,c]).div(avg_metrics.loc[:,c])
                except:
                    pass

            avg_metrics_deviation = avg_metrics_deviation.replace([np.inf, -np.inf, np.nan], 0)
            new_metrics[p] = avg_metrics_deviation
            
        
    return new_metrics

#combine all of the calculated metrics

def combine_data(data_dict):
    
    big_df = pd.DataFrame()
    for i in data_dict.keys():
        for j in data_dict[i].keys():
            big_df = pd.concat([big_df, data_dict[i][j].drop('DATE',axis=1)],axis=1)
    big_df['DATE'] = data_dict[i][j]['DATE']
    
    big_df['DATE'] = pd.to_datetime(big_df['DATE'])
    big_df = big_df[big_df['DATE']>'2012-01-01']


    return big_df

# check the model results. Period mins and max is still in progress (aka broke). Same with cumm hold and cumm performance

def check_model(predicted, returns_df, mins, maxs, results_column,returns_column,days_hold,buy_trigger,sell_trigger,leverage,trans_fee, margin_fee):
    
    model_result = pd.DataFrame()
    model_result = pd.merge(predicted[results_column],returns_df[returns_column],how='inner',left_index=True,right_index=True)

    model_result['holding result'] = returns_df[returns_column].div(days_hold).add(1)
    model_result['cumm hold'] = model_result['holding result'].cumprod()
    model_result['buy signal'] = model_result[results_column].apply(lambda x: 1 if x>buy_trigger else 0)
    model_result['sell signal'] = model_result[results_column].apply(lambda x: 1 if x<sell_trigger else 0)
    
    model_result['performance'] = 1
    model_result['max period loss'] = mins * leverage
    model_result['max period gain'] = maxs * leverage
    if leverage>1:
        margin_factor = 1
    else:
        margin_factor = 0

    for i, j in enumerate(model_result.iloc[:(-1-days_hold),4]):

        if j == 1:

            model_result.iloc[(i+days_hold-1),6] = ((model_result.iloc[i,1])*leverage/days_hold + 1) - margin_fee*leverage/days_hold*margin_factor
            if model_result.iloc[(i-1),4] != 1:
                model_result.iloc[(i+days_hold-1),6] = model_result.iloc[i,6] - trans_fee*leverage/days_hold

        else:
            model_result.iloc[(i+days_hold-1),6] = 1
            if model_result.iloc[(i-1),4] != 0:
                model_result.iloc[(i+days_hold-1),6] = model_result.iloc[i,6] - trans_fee*leverage/days_hold

            
    for i, j in enumerate(model_result.iloc[:(-1-days_hold),5]):
        if j == 1:
            model_result.iloc[(i+days_hold-1),6] = (model_result.iloc[i,1]*leverage*-1/days_hold + 1) - margin_fee*leverage/days_hold
            if model_result.iloc[(i-1),5] != 1:
                model_result.iloc[(i+days_hold-1),6] = model_result.iloc[i,6] - trans_fee*leverage/days_hold

        else:
            if model_result.iloc[(i-1),5] != 0:
               model_result.iloc[(i+days_hold-1),6] = model_result.iloc[i,6] - trans_fee*leverage/days_hold

   
    model_result['cumm performance'] = model_result['performance'].cumprod()
    plt.figure()
    model_result['cumm hold'].plot(legend=True, stacked=False)
    model_result['cumm performance'].plot(legend=True, stacked=False)

    
    for i in range(0,(days_hold)):
        port = 'Port ' + str(i)
        model_result[port] = 1
        for j in range(0,(len(model_result['performance'])-days_hold-1)):
            if (j % days_hold == i and model_result.iloc[j,4] == 1):
                model_result.iloc[j,10+i*2] = (model_result.iloc[(j+days_hold-1),6] + (margin_fee*leverage/days_hold*margin_factor) - 1) * days_hold + 1 - margin_fee*margin_factor - trans_fee*leverage*int(not bool(model_result.iloc[(j-1),4]-1))
            if (j % days_hold == i and model_result.iloc[j,5] == 1):
                model_result.iloc[j,10+i*2] = (model_result.iloc[(j+days_hold-1),6] + (margin_fee*leverage/days_hold) - 1) * days_hold - margin_fee + 1 - trans_fee*leverage*int(not bool(model_result.iloc[(j-1),5]-1))
        model_result[port + ' cumm'] = model_result[port].cumprod()        
        model_result[port + ' cumm'].plot(stacked=False)
        
    plt.figure()
    for i in range(0,(days_hold)):
        model_result['holding result'] = 1
        model_result['cumm hold'] = 1
        for j in range(0,(len(model_result['performance'])-days_hold-1)):
            if j % days_hold == i:
                model_result.iloc[j+days_hold-1,2] = model_result.iloc[j,1] + 1
        model_result['cumm hold'] = model_result['holding result'].cumprod() 
        model_result['cumm hold'].plot(title='hold')
        print(model_result.iloc[j,3])


    return model_result    


#%% Get the data
#frequency interval: 10m, 1h, 24h, 1w, 1month
period = '24h'
since = 	'1451628000'
#'s':since,

api_endpoints= pd.read_csv(r'CSV Location')

API_KEY = 'API Key'

res = requests.get('https://api.glassnode.com/v1/metrics/addresses/count',
    params={'a': 'BTC','s': since, 'i':period, 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])



URL_Start = 'https://api.glassnode.com'

metrics = pd.DataFrame
metrics = pd.DataFrame(df['t'])

for i,j in enumerate(api_endpoints['path']):
    current = pd.DataFrame()

    try:
        res = requests.get(URL_Start + api_endpoints['path'][i],
            params={'a': 'BTC', 'i':period, 'api_key': API_KEY})
        jsondata=json.loads(res.text)
        current=pd.DataFrame(jsondata)
        current['t'] = pd.to_datetime(df['t'], unit='s')
        current[j] = current['v']
        current.drop(columns='v',inplace=True)
        metrics = pd.merge(metrics,current,on=['t'],how='outer')
    except:
        pass

#
folder = 'Location Folder'
url = folder + period + "_metrics.csv"
metrics.to_csv(url)


res = requests.get('https://api.glassnode.com/v1/metrics/market/price_usd_ohlc',
    params={'a': 'BTC','s':since, 'i':period, 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])
ohlc = pd.DataFrame.from_records(df['o'], columns =['o', 'h', 'l','c'])
ohlc['Close'] = ohlc['c']
ohlc['t'] = df['t']
df = ohlc
df['DATE'] = df['t']
df.drop('t',axis=1,inplace=True)
df['DATE'] = pd.to_datetime(df['DATE'])

btc_data = df


metrics = metrics.replace([np.inf, -np.inf,np.nan], pd.NA)
metrics['DATE'] = metrics['t']
metrics['DATE'] = pd.to_datetime(metrics['DATE'])
metrics.drop(['t'],axis=1, inplace=True)
# columns 15,16,17 have data that gets too big and converts into a different format
try:
    metrics.drop(['/v1/metrics/mining/difficulty_mean','/v1/metrics/mining/difficulty_latest','/v1/metrics/mining/hash_rate_mean'],axis=1,inplace=True)
except:
    pass

original_data = pd.merge(metrics, btc_data, on='DATE', how='inner')

#%% Shape metrics
today_metrics = metrics[metrics['DATE']>'2021-01-01'] 

today_price_data = btc_data[btc_data['DATE']>'2021-01-01']   
#
period_dict = {'1- Fut': 1,
               '3- Fut': 3,
               '6- Fut': 6,
               '9- Fut': 9,
               '12- Fut': 12,
               '15- Fut': 15,
               '20- Fut': 20,
               '30- Fut': 30,
               '45- Fut': 45,
               '60- Fut': 60,
               '90- Fut': 90,
               '120- Fut': 120,
               '150- Fut': 150,
               '1- Past': -1,
               '3- Past': -3,
               '6- Past': -6,
               '9- Past': -9,
               '12- Past': -12,
               '15- Past': -15,
               '20- Past': -20,
               '30- Past': -30,
               '45- Past': -45,
               '60- Past': -60,
               '90- Past': -90,
               '120- Past': -120,
               '150- Past': -150,
               }

price_data_column = 'Close'
date_column = 'DATE'
min_data_column = 'l'
max_data_column = 'h'
#  
# Calculate future and past returns of the given time period
returns = lead_lag_return(original_data, period_dict, price_data_column,date_column)

#Calculate future and past minimum and max value of the given time period
#

price_mins = lead_lag_min_price(original_data, period_dict, min_data_column, date_column)
price_max = lead_lag_max_price(original_data, period_dict, max_data_column, date_column)

# calculate deviations from average for metrics

metric_period_dict = {
               '3- Past': 3,
               '6- Past': 6,
               '9- Past': 9,
               '12- Past': 12,
               '15- Past': 15,
               '20- Past': 20,
               '30- Past': 30,
               '45- Past': 45,
               '60- Past': 60,
               '90- Past': 90,
               '120- Past': 120,
               '150- Past': 150,
               }
metric_deviations_from_averages = average_deviation(metrics,metric_period_dict, 'Value')

metric_change = metrics.drop('DATE',axis=1).pct_change()
metric_change['DATE'] = metrics['DATE']

metric_deviations_from_average_change = average_deviation(metric_change,metric_period_dict, 'Avg Change')

# Combine the calculated metrics

metric_dict = {"Metric Deviation From Average": metric_deviations_from_averages,
              "Metric Deviation From Average Change": metric_deviations_from_average_change
              }
#

all_period_metrics = combine_data(metric_dict)

# Put Everything together

all_dataframes = [metric_change, all_period_metrics, btc_data,returns]

all_together_now = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],how='inner'), all_dataframes)


#
# Dimensionality Reduction


#Get rid of variables missing >20% data

# saving missing values in a variable
missing_data = all_together_now.isnull().sum()/len(all_together_now)*100
# saving column names in a variable
variables = all_together_now.columns
variable = [ ]
for i in range(0, len(variables)):
    if missing_data[i]<=20:   #setting the threshold as 20%
        variable.append(variables[i])
working = pd.DataFrame()
for i in range(0,len(variable)):
    working[variable[i]] = all_together_now[variable[i]]

working = working.replace([np.inf,-np.inf],np.nan).fillna(0)

missing_data = working == False
missing_data = missing_data.sum()/len(working)*100

variables = working.columns
variable = [ ]
for i in range(0, len(variables)):
    if missing_data[i]<=20:   #setting the threshold as 20%
        variable.append(variables[i])
working = pd.DataFrame()
for i in range(0,len(variable)):
    working[variable[i]] = all_together_now[variable[i]]



#
everything_reduced = working

results_tracker = {}
#%% Run Regressions and Test
split = '2020-06'
date_column = 'DATE'
return_column = '1- Fut Return'
days_hold = 1


everything_reduced = everything_reduced.replace([np.inf,-np.inf],np.nan).fillna(0)
X_train = everything_reduced[everything_reduced[date_column]<split].iloc[:,:-34].drop('DATE',axis=1)    
Y_train = everything_reduced[everything_reduced[date_column]<split][return_column]   
X_test = everything_reduced[everything_reduced[date_column]>=split].iloc[:,:-34].drop('DATE',axis=1)            
Y_test = everything_reduced[everything_reduced[date_column]>=split][return_column]    

#
OLS_fit_char = sm.OLS(Y_train,sm.add_constant(X_train)).fit()
OLS_pre_IS = OLS_fit_char.predict(sm.add_constant(X_train))
OLS_pre_OOS = OLS_fit_char.predict(sm.add_constant(X_test))
OLS_pre_IS = pd.DataFrame({'Fit':OLS_pre_IS})
OLS_pre_OOS = pd.DataFrame({'Fit':OLS_pre_OOS})

OLS_fit_char.summary()




# PCA example



# PCA applied to train and test data .95: .228
# Define data

# Fit PCA to train data
X_train_tran = StandardScaler().fit_transform(X_train)   # standardized the test
pca_model = PCA(.98) 
train_pca = pca_model.fit(X_train_tran)
fit_train_pca = pd.DataFrame(data =  pca_model.fit_transform(X_train_tran),index=X_train.index)

# Estimate OLS on train data with PCA factors
OLS_pca_fit = sm.OLS(Y_train,fit_train_pca).fit()
print(OLS_pca_fit.summary())
OLS_pca_pre_IS = OLS_pca_fit.predict(fit_train_pca)
OLS_pca_pre_IS = pd.DataFrame({'Fit':OLS_pca_pre_IS})

#
# Apply the train data PCA to test data
X_test_tran = StandardScaler().fit_transform(X_test)
fit_test_pca = pd.DataFrame(data =  pca_model.transform(X_test_tran),index=X_test.index)
OLS_pca_pre_OOS = OLS_pca_fit.predict(fit_test_pca)
OLS_pca_pre_OOS = pd.DataFrame({'Fit':OLS_pca_pre_OOS})
#



# Classification Model & Random Forest

# classification model

temp_model = RandomForestRegressor(n_estimators = 2000, n_jobs=-1, max_features = 200)
rf_fit = temp_model.fit(X=fit_train_pca,y=Y_train)

RF_pre_OOS = pd.DataFrame({'Fit':rf_fit.predict(fit_test_pca)},index=Y_test.index)




#


# Get numerical feature importances
importances = list(temp_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(everything_reduced.drop('DATE',axis=1).columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#

#




#
from matplotlib import colors
n_bins = 100

x = RF_pre_OOS['Fit']
y = Y_test

figs, axs = plt.subplots(1,2,sharey=True,tight_layout=True)

axs[0].hist(x,bins=n_bins)

axs[1].hist(y,bins=n_bins)

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=100, norm=colors.LogNorm())

return_check = pd.DataFrame()
return_check['Fit'] = x
return_check[return_column] = y


#
return_check['correct'] = 0
return_check['incorrect'] = 0
return_check['up'] = 0
return_check['down'] = 0 

for i,j in enumerate(return_check['Fit']):
    if float(j) > 0:
        return_check.iloc[i,4] = 1
    else:
        return_check.iloc[i,5] = 1
    if (float(j) > 0 and return_check.iloc[i,1] > 0):
        return_check.iloc[i,2] = 1
    elif (float(j)>0 and return_check.iloc[i,1]<0):
        return_check.iloc[i,3] = 1
    if (float(j) < 0 and return_check.iloc[i,1] < 0):
        return_check.iloc[i,2] = 1
    elif (float(j)<0 and return_check.iloc[i,1] > 0):
        return_check.iloc[i,3] = 1
        
return_check['correct'].sum()/(return_check['incorrect'].sum()+return_check['correct'].sum())
return_check['up'].sum() / (return_check['up'].sum() + return_check['down'].sum())

figs, axs = plt.subplots(1,2,sharey=True,tight_layout=True)
#


return_check['how correct'] = return_check['correct']*return_check[return_column]

return_check['how incorrect'] = return_check['incorrect']*return_check[return_column]

x = list(return_check['how correct'])

y = list(return_check['how incorrect'])

figs, axs = plt.subplots(1,2,sharey=True,tight_layout=True)
axs[0].hist(x,bins=n_bins)

axs[1].hist(y,bins=n_bins)

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=100, norm=colors.LogNorm())

#

correct_up = return_check['how correct'].apply(lambda x: 1 if x>0 else 0)
incorrect_up = return_check['how incorrect'].apply(lambda x: 1 if x<0 else 0)
correct_down = return_check['how correct'].apply(lambda x: 1 if x<0 else 0)
incorrect_down = return_check['how incorrect'].apply(lambda x: 1 if x>0 else 0)

return_check['correct up'] = correct_up
return_check['incorrect up'] = incorrect_up
return_check['correct down'] = correct_down
return_check['incorrect down'] = incorrect_down

correct_up_percent_total = return_check['correct up'].mean()
incorrect_up_percent_total = return_check['incorrect up'].mean()
correct_down_percent_total = return_check['correct down'].mean()
incorrect_down_percent_total = return_check['incorrect down'].mean()

correct_up_percent_predicted_up = correct_up.sum() / (correct_up.sum() + incorrect_up.sum())
correct_down_percent_predicted_down = correct_down.sum() / (correct_down.sum() + incorrect_down.sum())

ups = return_check['correct up'].sum() + return_check['incorrect down'].sum()

downs = return_check['incorrect up'].sum() + return_check['correct down'].sum()

up_percent = ups / (ups + downs)

down_percent = downs / (ups + downs)

z_score = st.norm.ppf(.95)

ce_cupt = 2*z_score*((correct_up_percent_total*(1-correct_up_percent_total))**.5 / ((ups+downs)**.5))
ce_iupt = 2*z_score*((incorrect_up_percent_total*(1-incorrect_up_percent_total))**.5 / ((ups+downs)**.5))
ce_cdpt = 2*z_score*((correct_down_percent_total*(1-correct_down_percent_total))**.5 / ((ups+downs)**.5))
ce_idpt = 2*z_score*((incorrect_down_percent_total*(1-incorrect_down_percent_total))**.5 / ((ups+downs)**.5))

ce_cuppu= 2*z_score*((correct_up_percent_predicted_up*(1-correct_up_percent_predicted_up))**.5 / ((ups)**.5))
ce_cdppd = 2*z_score*((correct_down_percent_predicted_down*(1-correct_down_percent_predicted_down))**.5 / ((downs)**.5))

arrays = [
    ["Correct up %", "Incorrect Up %","Correct Down %","Incorrect Down %"],
    [correct_up_percent_total,incorrect_up_percent_total, correct_down_percent_total,incorrect_down_percent_total ],
    ["CUPT CI", "IUPT CI, CDPT CI, IDPT CI"]
    [ce_cupt, ce_iupt, ce_cdpt, ce_idpt],
    ["Correct Up Prediction", "Correct Down Prediction"],
    [correct_up_percent_predicted_up, correct_down_percent_predicted_down],
    ["CUP Confidence Interval","CDP Confidence Interval"],
    [ce_cuppu, ce_cdppd],
    ["Actual % Up", "Actual % Down"],
    [up_percent, down_percent]
     ]


print(ce_cupt)
print('correct up ' + str(return_check['correct up'].sum()) + " " + str(return_check['correct up'].sum()/(return_check['correct up'].sum()+return_check['incorrect up'].sum())) + " Average: " + str(return_check['correct up'].mean()))
print('incorrect up ' + str(return_check['incorrect up'].sum()) + " " + str(return_check['incorrect up'].sum()/(return_check['correct up'].sum()+return_check['incorrect up'].sum())) + " Average: " + str(return_check['incorrect up'].mean()))
print('correct down ' + str(return_check['correct down'].sum()) + " " + str(return_check['correct down'].sum()/(return_check['correct down'].sum()+return_check['incorrect down'].sum())) + " Average: " + str(return_check['correct down'].mean()))
print('incorrect down ' + str(return_check['incorrect down'].sum()) + " " + str(return_check['incorrect down'].sum()/(return_check['correct down'].sum()+return_check['incorrect down'].sum())) + " Average: " + str(return_check['incorrect down'].mean()))

print('up times ' + str(return_check['up'].sum()) + " " + str(return_check['up'].sum() / (return_check['up'].sum() + return_check['down'].sum())))
print('down times ' + str(return_check['down'].sum()) + " " + str(return_check['down'].sum() / (return_check['up'].sum() + return_check['down'].sum())))

print(" Correct: " + str(return_check['correct up'].mean() + return_check['correct down'].mean()))

percent_correct = return_check['correct up'].mean() + return_check['correct down'].mean()

results_tracker[return_column] = percent_correct

win32api.MessageBox(0, 'Done. Results in Arrays Variable', 'Finished', 0x00001000) 


#%% Backtest the model on the test data
buy_trigger=0.1
sell_trigger=0
leverage=3
transaction_fee=.001
margin_fee = .0003
#

model_result_OLS = check_model(OLS_pre_OOS, everything_reduced, price_mins['1- Fut Price Min'], price_max['1- Fut Price Min'], 'Fit',return_column,days_hold,buy_trigger,sell_trigger,leverage,transaction_fee,margin_fee)

print(model_result_OLS.describe())

model_result_PCA = check_model(OLS_pca_pre_OOS, everything_reduced, price_mins['1- Fut Price Min'], price_max['1- Fut Price Min'], 'Fit',return_column,days_hold,buy_trigger,sell_trigger,leverage,transaction_fee,margin_fee)

print(model_result_PCA.describe())

model_result_RF = check_model(RF_pre_OOS, everything_reduced, price_mins['1- Fut Price Min'], price_max['1- Fut Price Min'], 'Fit',return_column,days_hold,buy_trigger,sell_trigger,leverage,transaction_fee,margin_fee)

print(model_result_RF.describe())
