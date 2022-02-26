# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:34:33 2022

@author: Dayanand
"""

# import library

import os
import pandas as pd
import numpy as np
import seaborn as sns

# Loading relevant TimeSeries library

from statsmodels.tsa.api import SimpleExpSmoothing,Holt,ExponentialSmoothing

# setting display size
pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",500)

# changing directory
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\DataSets")

# loading file
rawData=pd.read_csv("SuperStore.csv",encoding="ISO-8859-1")

#####
# Check for NA Values
#####

rawData.isna().sum()# we do not have NULL values

##### 
# Our column of interest is Sales & Date.So roll it Sales data on daily basis
# Let us sum up Sales column using Order Date columns

fullDf=rawData.groupby("Order Date")["Sales"].sum().reset_index().copy()

# Idealy we should have 365*4=1460 rows.But we have 889 rows only.It implies that our sales
# data is missing.And in timeseries we can not have missing values.So we require it to further 
# roll up the data.

# Before rolling up further let us convert all Data into one format

# Convert Date column into proper Date format

fullDf["Order Date"]=pd.to_datetime(fullDf["Order Date"])

# let us sort the date
fullDf.sort_values(["Order Date"],inplace=True)

# let us set date columns as index column.it will help us in plotting
fullDf.set_index(["Order Date"],inplace=True)

# let us min & max date
fullDf.index.min()
fullDf.index.max()

#  Further ROll Up
# This data is on daily basis.And in this data we have lots of missing values.
# In the time series we can not have missing datas. So it is advisable to further roll up in
# higher level like week or month
# we will roll up this data on monthly roll up

# we have resample function.We can roll up using this function
fullDf2=fullDf["Sales"].resample("MS").mean() # "MS-MonthStart" 
# We have rolled up month wise.So we will get total 12*4=48 rows of data.
# We have many other arguements also to roll up.We can also roll up on Sum/mean.
# But since mean gives less variation.So better to use mean function

# Plot the data
sns.lineplot(data=fullDf2)

# using the graph what we can infer is that we have trend as well as seasonal component.
# trend is not much visible clearly.

# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
Decompose_Series=seasonal_decompose(fullDf2)
Decompose_Series.plot()

# Sampling train & test split
train=fullDf2[:36].copy()
test=fullDf2[36:].copy()
train.shape
test.shape

##### Model building
# SES Model
#####

SES_Model=SimpleExpSmoothing(train).fit(smoothing_level=0.01)# model building
SES_Model.summary()

# Forescast the model
Forecast=SES_Model.forecast(12).rename("Forecast")
Actual_ForeCast_Df=pd.concat([fullDf2,Forecast],axis=1)

# plot
sns.lineplot(data=Actual_ForeCast_Df)

# Validation
ValidationDf=Actual_ForeCast_Df[-12:].copy()
#MAPE
np.mean(abs(ValidationDf["Forecast"]-ValidationDf["Sales"])/ValidationDf["Sales"])*100
# 37.76
#RMSE
np.sqrt(np.mean((ValidationDf["Forecast"]-ValidationDf["Sales"])**2))
# 303.47

# Double Exponential Model
# DES
DES_Model=Holt(train).fit(smoothing_level=0.01,smoothing_slope=0.06)
DES_Model.summary()

# Forecast the model
DES_Forecast=DES_Model.forecast(12).rename("Forecast")
Actual_Forecast_DES=pd.concat([fullDf2,DES_Forecast],axis=1)

# Plot
sns.lineplot(data=Actual_Forecast_DES)

# Validate the model
ValidationDES=Actual_Forecast_DES[-12:].copy()
#MAPE
np.mean(abs(ValidationDES["Forecast"]-ValidationDf["Sales"])/ValidationDf["Sales"])*100
#53.2
#RMSE
np.sqrt(np.mean((ValidationDES["Forecast"]-ValidationDES["Sales"])**2))
# 344.612

# Triple Exponential Model
# TES

TES_Model=ExponentialSmoothing(train,
                               trend="add",
                               seasonal="add",
                               seasonal_periods=12).fit(smoothing_level=0.01,
                                          smoothing_slope=0.03,
                                          smoothing_seasonal=0.01)
TES_Model.summary()

# Forecast the model
ForecastTES=TES_Model.forecast(12).rename("Forecast")
Actual_Forecast_TES=pd.concat([fullDf2,ForecastTES],axis=1)

# plot the model
sns.lineplot(data=Actual_Forecast_TES)

# Validate the model
ValidationTES=Actual_Forecast_TES[-12:].copy()
# MAPE
np.mean(abs(ValidationTES["Forecast"]-ValidationTES["Sales"])/ValidationTES["Sales"])*100
# 26.33
#RMSE
np.sqrt(np.mean((ValidationTES["Forecast"]-ValidationTES["Sales"])**2))
#185.741

# Automatic selection of smoothing, alpha,beeta & gaama
TES_Auto_Model=ExponentialSmoothing(train,seasonal_periods=12,
                                    trend="add",
                                    seasonal="add").fit()

TES_Auto_Model.summary()

# forecast the model
ForecastTESAuto=TES_Auto_Model.forecast(12).rename("Forecast")
Actual_ForecastTESAuto=pd.concat([fullDf2,ForecastTESAuto],axis=1)

# Plot
sns.lineplot(data=Actual_ForecastTESAuto)

# Validate
ValidationTESAuto=Actual_ForecastTESAuto[-12:]
#MAPE
np.mean(abs(ValidationTESAuto["Forecast"]-ValidationTESAuto["Sales"])/ValidationTESAuto["Sales"])*100
# 25.231
#RMSE
np.sqrt(np.mean((ValidationTESAuto["Forecast"]-ValidationTESAuto["Sales"])**2))
#177.69

### TES Manual Grid Search

import numpy as np

myAlpha=np.round(np.arange(0,1.1,0.1),2)
myBeta=np.round(np.arange(0,1.1,0.1),2)
myGamma=np.round(np.arange(0,1.1,0.1),2)

alphaList=[]
betaList=[]
gammaList=[]
mapeList=[]

for alpha in myAlpha:
    for beta in myBeta:
        for gamma in myGamma:
            
            TES=ExponentialSmoothing(train,
                                     seasonal_periods=12,
                                     seasonal="mul",
                                     trend="add").fit(smoothing_level=alpha,
                                                      smoothing_slope=beta,
                                                      smoothing_seasonal=gamma)
            
            Forecast=TES.forecast(12).rename("Forecast")
            Actual_Forecast_Df=pd.concat([fullDf2,Forecast],axis=1)
            ValidationDf=Actual_Forecast_Df[-12:].copy()
            tempMape=np.mean(abs(ValidationDf["Sales"]-ValidationDf["Forecast"])/ValidationDf["Sales"])*100
            alphaList.append(alpha)                                          
            betaList.append(beta)
            gammaList.append(gamma)
            mapeList.append(tempMape)
            
evaluationDf=pd.DataFrame({"alpha":alphaList,
                          "beta":betaList,
                          "gamma":gammaList,
                          "mape":mapeList
                          })

#######
##### ARIMA
######

from pmdarima.arima import ARIMA

### Model
arimaModel=ARIMA((2,1,0),(1,1,0,12)).fit(train)                          
Forecast=pd.Series(arimaModel.predict(12)).rename("Forecast")    
Forecast.index=test.index
Actual_Forecast_Df=pd.concat([fullDf2,Forecast],axis=1)

### plot
sns.lineplot(data=Actual_Forecast_Df)

## Validation
Validation_Df=Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df["Sales"]-Validation_Df["Forecast"])/Validation_Df["Sales"])*100
np.sqrt(np.mean((ValidationDf["Sales"]-Validation_Df["Forecast"])**2))

#####
### ARIMA AUTOMATED MODEL
####

from pmdarima import auto_arima
arimaModel2=auto_arima(train,m=12)

arimaModel2.get_params()["order"]
arimaModel2.get_params()["seasonal_order"]

#### Forecasting
Validation_Df=Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df["Sales"]-Validation_Df["Forecast"])/Validation_Df["Sales"])*100
np.sqrt(np.mean((ValidationDf["Sales"]-Validation_Df["Forecast"])**2))

### Adding some parameters like train=stationary and dagta transforamtion
sns.lineplot(data=train)

trainLog=np.log(train)
sns.lineplot(data=trainLog)

## Model
## if you believe that data is stationary meant trend is not visible
# then keep stationary=True

arimaModel3=auto_arima(trainLog,m=12,stationary=True)

## get the order of p,d,q,P,D,Q
arimaModel3.get_params()["order"]
arimaModel3.get_params()["seasonal_order"]

 ## Forecasting
Forecast=pd.Series(arimaModel3.predict(12)).rename("Forecast")
Forecast=np.exp(Forecast)
Forecast.index=test.index
Actual_Forecast_Df=pd.concat([fullDf2,Forecast],axis=1)

## Validation
sns.lineplot(data=Actual_Forecast)

np.mean(abs(Validation_Df["Sales"]-Validation_Df["Forecast"])/ValidationDf["Sales"])*100

### Grid Search
p=range(0,2)
d=range(0,2)
q=range(0,2)
P=range(2)
D=range(2)
Q=range(2)
myp=[]
myd=[]
myq=[]
myP=[]
myD=[]
myQ=[]
myMape=[]

for i in p:
    for j in d:
        for k in q:
            for I in P:
                for J in D:
                    for K in Q:
                        tempArima=ARIMA((i,j,k),(I,J,K,12)).fit(train)
                        Prediction=pd.Series(tempArima.predict(12)).rename("Forecast")
                        Forecast.index=test.index
                        Actual_Forecast_Df=pd.concat([fullDf2,Forecast],axis=1)
                        ValidationDf=Actual_Forecast_Df[-12:].copy()
                        tempMape=np.mean(abs(ValidationDf["Sales"]-ValidationDf["Forecast"])/ValidationDf["Sales"])*100
                        myp.append(i)
                        myd.append(j)
                        myq.append(k)
                        myP.append(I)
                        myD.append(J)
                        myQ.append(K)
                        myMape.append(tempMape)
                        
evaluationDf=pd.DataFrame({
    "p":myp,
    "d":myd,
    "q":myq,
    "P":myP,
    "D":myD,
    "Q":myQ,
    "MAPE":myMape})

## Finalize Arima full Model
arimaFullModel=ARIMA((0,1,1),(0,1,0,12)).fit(train)

## Forecasting
Forecast=pd.Series(arimaFullModel.predict(12)).rename("Forecast")
# Set he correct date in the index as per pediction 
start="2018-01-01" # check the order date
end="2018-12-01"
futureDateRange=pd.date_range(start,end,freq="MS")
Forecast.index=futureDateRange

Actual_Forecast_Df=pd.concat([fullDf2,Forecast],axis=1)

## Plot
sns.lineplot(data=Actual_Forecast_Df)

## Connected LinePLot
Acutal_Forecast_Series=pd.concat([fullDf2,Forecast],axis=0)
sns.lineplot(data=Acutal_Forecast_Series)
from matplotlib.pyplot import vlines
vlines(x=Acutal_Forecast_Series.index[48],ymin=0,ymax=max(Acutal_Forecast_Series),colors="red")
