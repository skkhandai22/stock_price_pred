import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

def closingtimeseries(stock):
    stock.index = pd.to_datetime(stock.index)
    monthly_mean = stock.Close.resample('M').mean()
    return monthly_mean

def decomposition(stock):
    monthly_mean = closingtimeseries(stock)
    decomposition = sm.tsa.seasonal_decompose(monthly_mean, model='additive')
    return decomposition

def plots(stock):
    stock.index = pd.to_datetime(stock.index)
    monthly_mean = stock.Close.resample('M').mean()

    decomposition = sm.tsa.seasonal_decompose(monthly_mean, model='additive')
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    l_param = []
    l_param_seasonal = []
    l_results_aic = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(monthly_mean,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                l_param.append(param)
                l_param_seasonal.append(param_seasonal)
                l_results_aic.append(results.aic)
            except:
                continue

    minimum=l_results_aic[0]
    for i in l_results_aic[1:]:
        if i < minimum:
            minimum = i
    i=l_results_aic.index(minimum)

    mod = sm.tsa.statespace.SARIMAX(monthly_mean,
                                    order=l_param[i],
                                    seasonal_order=l_param_seasonal[i],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)


    pred = results.get_prediction(start=pd.to_datetime('2017-12-31'), dynamic=False)
    pred_ci = pred.conf_int()

    y_forecasted = pred.predicted_mean
    y_truth = monthly_mean['2017-12-31':]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
 
    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    return monthly_mean ,decomposition,results,mod,pred,pred_uc,pred_ci

# def monthly_plot(stock):
#     stock.index = pd.to_datetime(stock.index)
#     monthly_mean = stock.Close.resample('M').mean()

#     #visualising closing time series data
#     return monthly_mean
#     monthly_mean.plot(figsize=(15, 6))
#     plt.show()

# def decomposition_plot(monthly_mean):
#     decomposition = sm.tsa.seasonal_decompose(monthly_mean, model='additive')
#     return decomposition

# def arima(monthly_mean):
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#     # print('Examples of parameter combinations for Seasonal ARIMA...')
#     # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#     # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#     # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#     # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#     l_param = []
#     l_param_seasonal = []
#     l_results_aic = []
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(monthly_mean,
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#                 results = mod.fit()
#                 # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#                 l_param.append(param)
#                 l_param_seasonal.append(param_seasonal)
#                 l_results_aic.append(results.aic)
#             except:
#                 continue

#     minimum=l_results_aic[0]
#     for i in l_results_aic[1:]:
#         if i < minimum:
#             minimum = i
#     i=l_results_aic.index(minimum)

#     mod = sm.tsa.statespace.SARIMAX(monthly_mean,
#                                     order=l_param[i],
#                                     seasonal_order=l_param_seasonal[i],
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)


#     return results,mod

# def forecast(results,monthly_mean):
#     pred = results.get_prediction(start=pd.to_datetime('2017-12-31'), dynamic=False)
#     pred_ci = pred.conf_int()

#     return pred,pred_ci

# def actual_pred(results,monthly_mean,pred):
    
#     # y_forecasted = pred.predicted_mean
#     # y_truth = monthly_mean['2017-12-31':]

#     # # Compute the mean square error
#     # mse = ((y_forecasted - y_truth) ** 2).mean()
 
#     # pred_uc = results.get_forecast(steps=100)
#     # pred_ci = pred_uc.conf_int()

#     # return pred_uc,pred_ci
#     ax = monthly_mean.plot(label='observed', figsize=(14, 7))
#     pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
#     ax.fill_between(pred_ci.index,
#                     pred_ci.iloc[:, 0],
#                     pred_ci.iloc[:, 1], color='k', alpha=.25)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('close price')

#     plt.legend()
#     plt.show()
    
