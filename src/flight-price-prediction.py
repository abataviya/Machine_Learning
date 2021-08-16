#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train_data= pd.read_excel("data/Data_Train.xlsx")
print(train_data.shape)
print(train_data.head())
pd.set_option('display.max_columns',None)
print(train_data.info())

print(train_data['Duration'].value_counts())
print(train_data.isnull().sum())
train_data.dropna(inplace=True)
print(train_data.isnull().sum())
print(train_data.shape)

# # EDA
train_data['journey_day']= pd.to_datetime(train_data.Date_of_Journey,format="%d/%m/%Y").dt.day
train_data['journey_month']= pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.month
print(train_data.head())
train_data.drop(['Date_of_Journey'], axis=1, inplace=True)
print(train_data.head())
train_data['Dep_hour']= pd.to_datetime(train_data.Dep_Time).dt.hour
train_data['Dep_min']= pd.to_datetime(train_data.Dep_Time).dt.minute
print(train_data.head())
train_data.drop(['Dep_Time'], axis=1, inplace= True)
print(train_data.head())

# # Converting arrival time to hour and minute
train_data['Arrival_hour']= pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data['Arrival_min']= pd.to_datetime(train_data.Arrival_Time).dt.minute
train_data.drop(['Arrival_Time'],axis=1, inplace=True)
print(train_data.head())
duration= list(train_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if "h" in duration[i]:
            duration[i]= duration[i].strip()+ ' 0m'
        else:
            duration[i]= '0h '+ duration[i]

duration_hours= []
duration_mins= []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep= 'm')[0].split()[-1]))
train_data['Duration_hours']= duration_hours
train_data['Duration_mins']= duration_mins
print(train_data.head())
train_data.drop(['Duration'], axis=1, inplace=True)
print(train_data.head(3))

print(train_data['Airline'].value_counts())
sns.catplot(y= 'Price', x='Airline', data= train_data.sort_values('Price', ascending= False), kind= 'boxen', height=6, aspect=3)
Airline= train_data[['Airline']]
Airline= pd.get_dummies(Airline, drop_first= True)
print(Airline.head())

print(train_data['Source'].value_counts())
sns.catplot(y= 'Price', x='Source', data= train_data.sort_values('Price', ascending= False), kind= 'boxen', height=6, aspect=3)
plt.show()
Source= train_data[['Source']]
Source= pd.get_dummies(Source, drop_first= True)
print(Source.head())

print(train_data['Destination'].value_counts())
sns.catplot(y= 'Price', x='Destination', data= train_data.sort_values('Price', ascending= False), kind= 'boxen', height=6, aspect=3)
plt.show()
Destination= train_data[['Destination']]
Destination= pd.get_dummies(Destination, drop_first= True)
print(Destination.head())
print(train_data.head(3))

train_data.drop(['Route','Additional_Info'], axis=1, inplace=True)
print(train_data.head())
print(train_data['Total_Stops'].value_counts())

# # As Total_Stops is ordinal data not nominal data so we will use LabelEncoder here
train_data.replace({'non-stop':0, '1 stop':1,'2 stops':2, '3 stops':3, '4 stops':4}, inplace= True)

# concatenate DataFrame (train_data + Airline + Source + Destination)
data_train= pd.concat([train_data, Airline, Source, Destination], axis=1)
print(data_train.head())
data_train.drop(['Airline','Source','Destination'], axis=1, inplace= True)
print(data_train.head())
print(data_train.shape)

# Test data EDA
test_data= pd.read_excel("data/Test_set.xlsx")
print(test_data.shape)
print(test_data.head())
