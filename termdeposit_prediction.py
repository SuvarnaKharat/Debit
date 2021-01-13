# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:18:42 2020

@author: admin
"""

#Load the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv file
data = pd.read_csv("C:/Users/admin/Desktop/python program files/Project/Project3/IIT NON PROC.csv")

#Exploratory Data Analysis

#Understand the data

data.head()

data.tail()

data.columns

data.shape
#5581,18

data.dtypes
#Unnamed: 0     int64
#age            int64
#job           object
#marital       object
#education     object
#default       object
#balance        int64
#housing       object
#loan          object
#contact       object
#day            int64
#month         object
#duration       int64
#campaign       int64
#pdays          int64
#previous       int64
#poutcome      object
#deposit       object

#There are total 18 columns and 5581 rows in the dataset.
#Target Variable is deposit (Yes/No)

data.info()
# 8 integer variables and 10 Categorical variables

summary_int =data.describe()
summary_cat = data.describe(include='O')

#Check for Null values :
    
data.isnull().sum()
#No NULL values in dataset

#Check for duplicate values:
    
data.duplicated().sum()
#No Duplicate values

#Check for unique values :
data.nunique()

#Check the unique values of each categorical variable:

data["job"].value_counts()  #12 unique values
data["marital"].value_counts()  #3 unique values
data["education"].value_counts()  #4 unique values
data["contact"].value_counts()  #3 unique values
data["month"].value_counts()  #12 unique values
data["poutcome"].value_counts()  #4 unique values    

#Data set is clean. There are no Null values and Duplicate values in dataset.

# Correlation between variables:
    
corr = data.corr()
corr
#Only previous and pdays are having good correlation

    
sns.pairplot(data)

# Correlation with Output Variable:

data.columns
#Index(['Unnamed: 0', 'age', 'job', 'marital', 'education', 'default',
#       'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration',
#       'campaign', 'pdays', 'previous', 'poutcome', 'deposit'],
#      dtype='object')

#job Vs deposit

job_dep = pd.crosstab(index=data["job"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(job_dep)

sns.countplot(data['job'],hue=data['deposit'])

#Clients with different jobs affects term deposit

#Marital Vs deposit

marital_dep = pd.crosstab(index=data["marital"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(marital_dep)

sns.countplot(data['marital'],hue=data['deposit'])

#Education Vs deposit

edu_dep = pd.crosstab(index=data["education"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(edu_dep)

sns.countplot(data['education'],hue=data['deposit'])

#default Vs deposit

def_dep = pd.crosstab(index=data["default"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(def_dep)

sns.countplot(data['default'],hue=data['deposit'])

#Imbalance data.
#important variable in analysis

hou_dep = pd.crosstab(index=data["housing"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(hou_dep)

sns.countplot(data['housing'],hue=data['deposit'])

#important variable in analysis

loan_dep = pd.crosstab(index=data["loan"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(loan_dep)

sns.countplot(data['loan'],hue=data['deposit'])

#Imbalance data.
#important variable in analysis

cont_dep = pd.crosstab(index=data["contact"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(cont_dep)

sns.countplot(data['contact'],hue=data['deposit'])

#Important variable in analysis

mon_dep = pd.crosstab(index=data["month"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(mon_dep)

sns.countplot(data['month'],hue=data['deposit'])

#Important variable in analysis


pout_dep = pd.crosstab(index=data["poutcome"],
                      columns=data["deposit"],
                      margins=True,
                      normalize='index')
print(pout_dep)

sns.countplot(data['poutcome'],hue=data['deposit'])

#Important variable in analysis

#Variables like Unnamed0, Marital, Education are not important

#Check the count of Output Variable:
sns.countplot(data['deposit'])
#The distribution looks balanced.


#Relationship b/w numerical variables with Output variable:
    
data.columns
#numerical variables: age, balance, day, duration, campaign, pdays, previous

# Age VS deposit
sns.boxplot('deposit','age',data=data)
data.groupby('deposit')['age'].median()
#peoople with age >39 have taken the deposit

#Balance VS deposit
sns.boxplot('deposit','balance',data=data)
data.groupby('deposit')['balance'].median()

# Day VS deposit
sns.boxplot('deposit','day',data=data)
data.groupby('deposit')['day'].median()

#Duration VS deposit
sns.boxplot('deposit','duration',data=data)
data.groupby('deposit')['duration'].median()
#duration affects deposit

#campaign VS deposit
sns.boxplot('deposit','campaign',data=data)
data.groupby('deposit')['campaign'].median()

#pdays VS deposit
sns.boxplot('deposit','pdays',data=data)
data.groupby('deposit')['pdays'].median()

#previous VS deposit
sns.boxplot('deposit','previous',data=data)
data.groupby('deposit')['previous'].median()

#Outlier Detection:
sns.boxplot('age',data=data)
sns.boxplot('balance',data=data)
sns.boxplot('day',data=data) #No Outlier
sns.boxplot('duration',data=data)
sns.boxplot('campaign',data=data)
sns.boxplot('pdays',data=data)
sns.boxplot('previous',data=data)

#Base model without cleaning the data:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data_copy = data.copy()

data_copy = data_copy.drop('Unnamed: 0',axis=1)
data_copy['deposit']=data_copy['deposit'].map({'yes':1,'no':0})
data_copy=pd.get_dummies(data_copy,drop_first=True)
data_copy.shape

#seperate I/P and O/P feature
x=data_copy.drop('deposit',axis=1)
x.shape

y=data_copy.loc[:,'deposit']

#Split the data into test and train

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#Model:
model=LogisticRegression()
model.fit(x_train,y_train)

pred= model.predict(x_test)
pred
conf_mat = confusion_matrix(y_test,pred)
print(conf_mat)

acc=accuracy_score(y_test,pred)
acc
#80% accuracy for base model


# Data Cleaning:
#Drop the irrelevent columns:
data.columns
data_m1 = data.copy()
data_m1=data_m1.drop(['Unnamed: 0','marital','education'],axis=1)
data_m1.columns
data_m1.shape

#Removing Outliers:

#'age''balance''duration''campaign''pdays''previous'
    
#By IQR method:

data_out = data_m1[['age','balance','duration','campaign','pdays','previous']]
data_out.columns.tolist()
            
for col in data_out.columns:
    percentiles = data_out[col].quantile([0.10,0.90]).values
    print(percentiles)
    data_out[col][data_out[col] <= percentiles[0]] = percentiles[0]
    data_out[col][data_out[col] >= percentiles[1]] = percentiles[1]

data_out.shape      #5581,6



#for val in data_out['balance'].tolist():
#   if val >= upper or val<=lower:
#       outlier.append(val)
#print(outlier)


#Check whether the Outlier are removed or not:
sns.boxplot('age',data=data_out)
sns.boxplot('balance',data=data_out)
sns.boxplot('duration',data=data_out)
sns.boxplot('campaign',data=data_out)
sns.boxplot('pdays',data=data_out)
sns.boxplot('previous',data=data_out)

data_m1['age']=data_out['age'] 
data_m1['balance']=data_out['balance'] 
data_m1['duration']=data_out['duration'] 
data_m1['campaign']=data_out['campaign'] 
data_m1['pdays']=data_out['pdays'] 
data_m1['previous']=data_out['previous'] 

data_m1.shape
data_m1.head()

#Model Building:
    
#1. Logistic Regression:

data_m1['deposit']=data_m1['deposit'].map({'yes':1,'no':0})
data_m1=pd.get_dummies(data_m1)
data_m1.shape

#seperate I/P and O/P feature
x=data_m1.drop('deposit',axis=1)
x.shape

y=data_m1.loc[:,'deposit']

#Split the data into test and train

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#Model:
model=LogisticRegression()
model.fit(x_train,y_train)

pred_m1= model.predict(x_test)
pred_m1
conf_mat_m1 = confusion_matrix(y_test,pred_m1)
print(conf_mat_m1)

acc_m1=accuracy_score(y_test,pred_m1)
acc_m1
#81.73% 

# K Nearest Neighbour:

from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import classification_report

knn = knc(n_neighbors =5)
knn.fit(x_train,y_train)

pred_m2 = knn.predict(x_test)
acc_m2 = accuracy_score(y_test,pred_m2)
acc_m2   #74.47%

clas_rep = classification_report(y_test,pred_m2)
clas_rep

#Standardization of data:
    
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#Train Dataset:
scalar.fit(x_train)
scaled_feat = scalar.transform(x_train)

x_train_std = pd.DataFrame(scaled_feat, columns=x_train.columns)
x_train_std.head()

#Test Dataset:

scalar.fit(x_test)
scaled_feat_t = scalar.transform(x_test)

x_test_std = pd.DataFrame(scaled_feat_t, columns=x_test.columns)
x_test_std.head()


#Apply K Nearest Neighbout Algo:

knn = knc(n_neighbors =5)
knn.fit(x_train_std,y_train)

pred_m2_s = knn.predict(x_test_std)
acc_m2_s = accuracy_score(y_test,pred_m2_s)
acc_m2_s   #78.38%

#Calculate the best K value:

error_rate = []
for i in range(1,40):
    knn = knc(n_neighbors =i)
    knn.fit(x_train_std,y_train)
    pred_i = knn.predict(x_test_std)
    error_rate.append(np.mean(pred_i!=y_test))
    acc_i = accuracy_score(y_test,pred_i)
    print(acc_i)
    
# i =1      0.7486567164179104
# i =2      0.7176119402985075
# i =4      0.7743283582089552
# i =5      0.7534328358208955
# i =6      0.7838805970149254
# i =7      0.7653731343283582
# i =8      0.7791044776119403
# i =9      0.764179104477612
# i =10      0.7767164179104478
# i =11      0.7629850746268657
# i =12      0.7713432835820896
# i =13      0.7665671641791045
# i =14      0.7737313432835821
# i =15      0.7701492537313432
# i =16      0.7695522388059701
# i =17      0.7701492537313432
# i =18      0.7785074626865671
# i =19      0.7719402985074627
# i =20      0.7785074626865671
# i =21      0.7737313432835821
# i =22      0.7743283582089552
# i =23      0.7725373134328358
# i =24      0.7785074626865671
# i =25      0.7755223880597015
# i =26      0.7820895522388059
# i =27      0.7725373134328358
# i =28      0.7785074626865671
# i =29      0.7647761194029851
# i =30      0.7773134328358209
# i =31      0.7635820895522388
# i =32      0.7701492537313432
# i =33      0.7671641791044777
# i =34      0.7797014925373135
# i =35      0.7731343283582089
# i =36      0.7773134328358209
# i =37      0.7677611940298508
# i =38      0.777910447761194
# i =39      0.7713432835820896
# i =40      0.7791044776119403


#model Without removing Outlier:
    
data_wo = data.copy()

data_wo['deposit']=data_wo['deposit'].map({'yes':1,'no':0})
data_wo=pd.get_dummies(data_wo)
data_wo.shape

#seperate I/P and O/P feature
x_wo=data_wo.drop('deposit',axis=1)
x_wo.shape

y_wo=data_wo.loc[:,'deposit']

#Split the data into test and train

x_train_wo,x_test_wo,y_train_wo,y_test_wo = train_test_split(x_wo,y_wo,test_size=0.3,random_state=0)

error_rate_wo = []
for i in range(1,25):
    knn = knc(n_neighbors =i)
    knn.fit(x_train_wo,y_train_wo)
    predwo_i = knn.predict(x_test_wo)
    error_rate_wo.append(np.mean(predwo_i!=y_test_wo))
    accwo_i = accuracy_score(y_test_wo,predwo_i)
    print(accwo_i)

error_rate_wo

#Not working accuracy = 70%

#Random forest classifier:
    
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
model3 =classifier.fit(x_train_std,y_train)
pred_rf = model3.predict(x_test_std)

conf_mat_rf = confusion_matrix(y_test,pred_rf)
conf_mat_rf

acc_rf = accuracy_score(y_test,pred_rf)
acc_rf   #84.47

clas_rep_rf = classification_report(y_test,pred_rf)


