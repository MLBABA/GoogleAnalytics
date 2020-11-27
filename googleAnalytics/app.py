#!/usr/bin/env python
# coding: utf-8




from flask import Flask, jsonify, request
import pandas as pd
from flask import  render_template
import numpy as np
from sklearn import linear_model
import joblib
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import re
import catboost
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


def function_1(data_point):
    
    columns_to_be_considered = ['channelGrouping', 'date', 'fullVisitorId', 'visitId', 'visitNumber',
       'visitStartTime', 'device.browser', 'device.operatingSystem',
       'device.isMobile', 'device.deviceCategory', 'geoNetwork.continent',
       'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region',
       'geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.networkDomain',
       'totals.hits', 'totals.pageviews', 'totals.timeOnSite',
       'totals.sessionQualityDim', 'totals.transactions',
       'totals.transactionRevenue', 'trafficSource.referralPath',
       'trafficSource.campaign', 'trafficSource.source',
       'trafficSource.medium', 'trafficSource.keyword',
       'trafficSource.adContent', 'weekday', 'day', 'month', 'year', 'visitHour', 'is_weekend']
    Data = pd.DataFrame(data_point,columns = columns_to_be_considered)
#     print(Data.columns)
    Data["date"] = pd.to_datetime(Data["date"])# seting the column as pandas datetime
    Data["weekday"] = Data['date'].dt.weekday #extra cting week day
    Data["day"] = Data['date'].dt.day # extracting day
    Data["month"] = Data['date'].dt.month # extracting month
    Data["year"] = Data['date'].dt.year # extracting year
    Data['visitHour'] = (Data['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int) #instaed of this we could also use (pd.datetime(df['visitstarttime'], unit = 's').dt.hour
    weekday_vals_t = Data['weekday'].values
    weekend_t = []
    for i in weekday_vals_t:
        if (i == 0) or (i == 6):
            weekend_t.append(1)
        else:
            weekend_t.append(0)
    Data['is_weekend'] = weekend_t
    
    #     missing values noww
    #     numerical values  
    numerical_float_features = ['visitNumber','visitStartTime','totals.hits','totals.pageviews',\
                'totals.timeOnSite','totals.transactions','totals.transactionRevenue']
    for i in numerical_float_features:
        Data[i].fillna(0,inplace=True)
        Data[i] = Data[i].astype('float')
    print("float done       ........................")

    
#     label_encoding
    categorical_feat = ['channelGrouping','device.browser','device.operatingSystem','device.deviceCategory',
                        'geoNetwork.continent','geoNetwork.subContinent','geoNetwork.country','geoNetwork.region',
                        'geoNetwork.metro','geoNetwork.city','geoNetwork.networkDomain',
                        'trafficSource.campaign','trafficSource.source','trafficSource.medium','trafficSource.keyword',
                        'trafficSource.referralPath', 'trafficSource.adContent']
    for feature in categorical_feat:
    
        label_encoder = preprocessing.LabelEncoder() 
        container = np.load(feature+'.npz')
        data = [container[key] for key in container]
        label_encoder.classes_ = data[0]     
        a += 1
        Data[feature]  = label_encoder.transform(list(Data[feature].values.astype('str')))
    
    print("categorical feature preprocessing done..!")
    


# label encoding end

    
    test_frame_k_maxdate = max(Data['date'])
    test_frame_k_mindate = min(Data['date'])
    print(test_frame_k_maxdate)
    print(test_frame_k_mindate)
    print(type(test_frame_k_maxdate))
    print(type(test_frame_k_mindate))
    
    test_frame_featurized = Data.groupby('fullVisitorId').agg({
            'geoNetwork.networkDomain': [('networkDomain' , lambda x: x.dropna().max())], #max value of network domain
            'geoNetwork.city':          [('city' , lambda x: x.dropna().max())],  #max value of city
            'device.operatingSystem':   [('operatingSystem' , lambda x: x.dropna().max())],  #max value of Operating System
            'geoNetwork.metro':         [('metro' , lambda x: x.dropna().max())],  #max value of metro
            'geoNetwork.region':        [('region' , lambda x: x.dropna().max())],   #max vaue of region
            'channelGrouping':          [('channelGrouping' , lambda x: x.dropna().max())],  #max value of channel grouping
          'trafficSource.referralPath': [('referralPath' , lambda x: x.dropna().max())],  #max value of referral path
            'geoNetwork.country':       [('country' , lambda x: x.dropna().max())],    #max value of country
            'trafficSource.source':     [('source' , lambda x: x.dropna().max())],   #max value of source
            'trafficSource.medium':     [('medium' , lambda x: x.dropna().max())],   #max value of medium
            'trafficSource.keyword':    [('keyword', lambda x: x.dropna().max())], #max value of keyboard
            'device.browser':           [('browser' , lambda x: x.dropna().max())],  #max value of browser
            'device.deviceCategory':    [('deviceCategory', lambda x: x.dropna().max())], #max of device category
            'geoNetwork.continent':     [('continent' , lambda x: x.dropna().max())],      #max of continent value
            'geoNetwork.subContinent':  [('subcontinent' , lambda x: x.dropna().max())],  #max of sub_continent value
            'totals.timeOnSite':        [('timeOnSite_sum'  , lambda x: x.dropna().sum()),     # total timeonsite of user
                                         ('timeOnSite_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('timeOnSite_max'  , lambda x: x.dropna().max()),     # max timeonsite
                                         ('timeOnSite_mean' , lambda x: x.dropna().mean())],  # mean timeonsite
            'weekday':                  [('weekday_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('weekday_max'  , lambda x: x.dropna().max())],  # mean timeonsite
            'day':                      [('day_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('day_max'  , lambda x: x.dropna().max())],  # mean timeonsite
            'month':                    [('month_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('month_max'  , lambda x: x.dropna().max())],  # mean timeonsite
            'year':                     [('year_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('year_max'  , lambda x: x.dropna().max())],  # mean timeonsite
            'visitHour':                [('visitHour_min'  , lambda x: x.dropna().min()),     # min timeonsite
                                         ('visitHour_max'  , lambda x: x.dropna().max())],  # mean timeonsite
            'totals.pageviews':         [('pageviews_sum'  , lambda x: x.dropna().sum()),     # total of page views
                                         ('pageviews_min'  , lambda x: x.dropna().min()),     # min of page views
                                         ('pageviews_max'  , lambda x: x.dropna().max()),     # max of page views
                                         ('pageviews_mean' , lambda x: x.dropna().mean())],  # mean of page views
            'totals.hits':              [('hits_sum'  , lambda x: x.dropna().sum()),     # total of hits
                                         ('hits_min'  , lambda x: x.dropna().min()),     # min of hits
                                         ('hits_max'  , lambda x: x.dropna().max()),     # max of hits
                                         ('hits_mean' , lambda x: x.dropna().mean())],  # mean of hits
            'visitStartTime':           [('visitStartTime_counts' , lambda x: x.dropna().count())], #Count of visitStartTime
            'totals.sessionQualityDim': [('sessionQualityDim' , lambda x: x.dropna().max())], #Max value of sessionQualityDim
            'device.isMobile':          [('isMobile' ,  lambda x: x.dropna().max())], #Max value of isMobile
            'visitNumber':              [('visitNumber_max' , lambda x: x.dropna().max())],  #Maximum number of visits.
            'totals.transactions' :     [('transactions' , lambda x:x.dropna().sum())], #Summation of all the transaction counts.
            'date':                     [('first_ses_from_the_period_start' , lambda x: x.dropna().min() - test_frame_k_mindate), #first shopping session for customer after the period end date for current frame.
                                         ('last_ses_from_the_period_end', lambda x: test_frame_k_maxdate - x.dropna().max()), #Last shopping session for customer before the period end date for current frame.
                                         ('interval_dates' , lambda x: x.dropna().max() - x.dropna().min()),  #interval calculated as the latest date on which customer visited - oldest date on which they visited.
                                         ('unqiue_date_num' , lambda x: len(set(x.dropna())))] , # Unique number of dates customer visited.           
                                                         })
    test_frame_featurized.columns = test_frame_featurized.columns.droplevel() 
    
    test_frame_featurized['interval_dates'] = test_frame_featurized['interval_dates'].dt.days
    test_frame_featurized['first_ses_from_the_period_start'] = test_frame_featurized['first_ses_from_the_period_start'].dt.days
    test_frame_featurized['last_ses_from_the_period_end'] = test_frame_featurized['last_ses_from_the_period_end'].dt.days
    
 
    test_frame_featurized = test_frame_featurized.reset_index()

    print("feature engineering process done..!")
    
    return test_frame_featurized
    
@app.route('/')
def hello_world():
    return 'Hello World!'


# In[4]:


@app.route('/index')
def index():
    return flask.render_template('index.html')


# In[6]:


@app.route('/fun1', methods=['POST'])
def fun1():
    RF_classification_model = joblib.load('classifier_catboost_19.pkl')
    data=[x for x in request.form.values()]
    data_encoded=function_1(data)
    
    classification_pred  = RF_classification_model.predict(data_encoded.drop('fullVisitorId', axis=1))     

    print("classoooo   doneeeee")

    RF_regression_model = joblib.load('Regressor_catboost_19.pkl') 
    
    regression_pred      = RF_regression_model.predict(data_encoded.drop('fullVisitorId', axis=1))
    
    print('Regression Done...................')
        
    final_prediction     =  classification_pred*regression_pred
    
    print("prediction for query point done..!")
    
    return final_prediction

    return render_template('index.html',prediction_text="Prediction is {}".format(prediction))


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




