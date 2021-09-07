#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 19:39:06 2021

@author: tracysheng
"""
#--------------------------------
#HW1_part3 Logistic
#--------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import scipy.signal
from scipy.interpolate import interp1d #,spline 
#-----------------------------------
#Goal:read in dataset
    
data=pd.read_json("/Users/tracysheng/590-CODES/DATA/weight.json")    
x=data.y
y=data.is_adult
   
#------------------------------------
#Goal:Normalize inputs (features) and outputs as needed using “standard scalar”

#Theory:Standardize features by removing the mean and scaling to unit variance
#Reason:Standardization is a common requirement for many machine learning estimators: they might behave badly if not standard normally distributed 

x_mean= sum(x)/len(x)
x_std= np.std(x)
x_stan = (x-x_mean)/x_std

y_mean= sum(y)/len(y)
y_std= np.std(y)
y_stan = (y-y_mean)/y_std

#---------------------------------------
#Goal: Deciding on an evaluation protocol as needed using 80% training and 20% test

X_train, X_valid, Y_train, Y_valid = train_test_split(x_stan,y_stan,test_size = 0.20)

#---------------------------------------
#Developing Logistic Regression
#1.Define a model function that takes a vector x and vector p.

def model(x,p):
        return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#2.Define an objective (loss) function

def loss(p):
    yp=model(X_train,p)
    MSE=(1/len(X_train))*sum((Y_train-yp)**2)
    return MSE

def loss_y(p):
    yp=model(X_valid,p)
    MSE=(1/len(X_valid))*sum((Y_valid-yp)**2)
    return MSE
    
#3. use scipy optimizer to minimize the loss function and obtain the optimal m,b parameters

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
po=np.random.uniform(0.5,1.,size=4)

#TRAIN MODEL USING SCIPY OPTIMIZER
res = minimize(loss, po, method='Nelder-Mead', tol=1e-15)
popt=res.x
print("OPTIMAL PARAM:",popt)


#Validation MODEL USING SCIPY OPTIMIZER
res_validation = minimize(loss_y, po, method='Nelder-Mead', tol=1e-15)
popt=res_validation.x
print("OPTIMAL PARAM:",popt)

#---------------------------------------------------
#SAVE HISTORY FOR PLOTTING AT THE END
train_iterations=[]
val_iterations=[]
loss_train=[]
loss_val=[]
train_iteration=0
val_iteration=0

def train_loss_history(p):
    global train_iteration, train_iterations, loss_train
    obj = loss(p)
    loss_train.append(obj)
    train_iterations.append(train_iteration)     
    train_iteration+=1
    
def validation_loss_history(p):
    global val_iteration,val_iterations,loss_val
    obj = loss_y(p)
    loss_val.append(obj)
    val_iterations.append(val_iteration)     
    val_iteration+=1

res = minimize(loss, po, method='Nelder-Mead', callback=train_loss_history,tol=1e-15)

res_validation = minimize(loss_y, po, method='Nelder-Mead', callback=validation_loss_history,tol=1e-15)

#------------------
##Visualize result
#plot1:train loss and validation loss
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(train_iterations, loss_train, color='blue',label='train loss')
ax.plot(val_iterations, loss_val, color='chartreuse', label='validation loss')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')

plt.xlabel("train_iterations")
plt.ylabel("loss")


#---------------------
#dstandardized the output

dstan_X_train = X_train * x_std + x_mean
dstan_X_valid = X_valid * x_std + x_mean
dstan_Y_train = Y_train * y_std + y_mean
dstan_Y_valid = Y_valid * y_std + y_mean

dstan_model= model(X_valid,popt) * y_std + y_mean

#GENERATE SMOOTH LINE THROUHG NOISY DATA

#SMOTHING METHOD-1: savgol_filter
window=5 #window must me odd and requires adjustment depending on plot

#SMOOTHED DATA (SAME NUMBER OF POINTS AS y)
#https://riptutorial.com/scipy/example/15878/using-a-savitzky-golay-filter
ys = scipy.signal.savgol_filter(dstan_model, window, 4)  # window size , polynomial order

# #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
xs1=np.linspace(min(dstan_X_valid), max(dstan_X_valid), int(0.25*len(dstan_X_valid)))
F=interp1d(dstan_X_valid, ys, kind='quadratic');
ys1=F(xs1); 
#-----------------------
#plot2 
fig, ax = plt.subplots()

ax.plot(dstan_X_train, dstan_Y_train, 'o',color='blue',label='train set')
ax.plot(dstan_X_valid, dstan_Y_valid ,'o', color='chartreuse', label='validation set')
ax.plot(xs1,ys1,color='black',linewidth=1.0,label="model(savgol smoothing)") 
ax.plot()

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
plt.xlabel("x")
plt.ylabel("y")

#-----------------------
#plot3 parity plot
#parity plot(plots the model predictions yp as a function of data y)

fig, ax = plt.subplots()
ax.plot(dstan_Y_valid, model(dstan_X_valid,popt), 'o',color='blue',label='train set')
plt.xlabel("Ground Truth")
plt.ylabel("Y prediction")

