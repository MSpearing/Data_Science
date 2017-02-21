
# coding: utf-8

# In[2]:

# Michael Spearing
# February 15, 2017
# Data Science Lab 5

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as skl_lm
import sklearn as skl
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:

# Chapter 3 Question 10
carseats = pd.read_csv('input/Carseats.csv')
carseats.info()


# In[ ]:

# A) Fit a multiple regression model to predict Sales using Price, Urban, and US
est = smf.ols('Sales ~ Price + Urban + US', carseats).fit()
est.summary()
carseats.corr()
regr = skl_lm.LinearRegression()

x = carseats[['Price', 'Urban', 'US']].as_matrix()
x[x=='Yes'] = 1
x[x=='No'] = 0
y = carseats.Sales
regr.fit(x,y)
print(regr.coef_)
print(regr.intercept_)


# In[67]:

# Chapter 4 Question 10
weekly = pd.read_csv('./input/weekly.csv')
weekly = weekly.iloc[:,1:]


# In[71]:

# A) Produce some numerical and graphical summaries of the Weekly 
# data. Do there appear to be any patterns?
weekly.info()
weekly.head()
print(weekly.corr())
scatter_matrix(weekly)
plt.show()
year_avg = [None] *(21)
for i in range(21):
    year_avg[i] = weekly[weekly['Year'] == (i + 1990)]
    year_avg[i] = year_avg[i]['Today']
    year_avg[i] = year_avg[i].mean()
plt.plot(range(1990 , 2011),year_avg)
plt.scatter(weekly['Year'],weekly['Today'])
plt.show()


# In[87]:

# B) Use the full data set to perform a logistic regression with 
# Direction as the response and the five lag variables plus Volume 
# as predictors. Use the summary function to print the results. Do 
# any of the predictors appear to be statistically significant? If so, 
# which ones?
model_LR = skl_lm.LogisticRegression()
model_LR = model_LR.fit(weekly.iloc[:,1:7], weekly['Direction'])
predicted_LR = model_LR.predict(weekly.iloc[:,1:7])
wrong = 0
for i in range(len(predicted_LR)):
    if(predicted_LR[i] != weekly.iloc[i,8]):
        wrong += 1
best = skl.feature_selection.SelectKBest(k='all')
best = best.fit(weekly.iloc[:,1:7], weekly['Direction'])
print "P-Values for Features Lag1 - Lag5 and Volume:"
print best.pvalues_
print("\nNumber of miss classifications: " + str(wrong))
print("\nCoeficients of predictors: \n" + str(model_LR.coef_))


# In[89]:

# C) Compute the confusion matrix and overall fraction of correct 
# predictions. Explain what the confusion matrix is telling you about 
# the types of mistakes made by logistic regression.
cm_LR = confusion_matrix(weekly['Direction'],predicted_LR, labels=['Up','Down'] )
print("Confusion Matrix: ")
print(cm_LR)
print("Fraction of Correct Predictions: " + str(cm_LR[0,0] + cm_LR[1,1]) + "/" + str(cm_LR.sum()))
print("Percent of Correct Predictions: " + str(((cm_LR[0,0] + cm_LR[1,1]) / float(cm_LR.sum()))*100))


# In[92]:

# D) Now fit the logistic regression model using a training data period 
# from 1990 to 2008, with Lag2 as the only predictor. Compute the 
# confusion matrix and the overall fraction of correct predictions for 
# the held out data (that is, the data from 2009 and 2010).
trainingData = weekly[weekly.Year < 2009]
X_td = trainingData['Lag2'].reshape(-1,1)
y_td = trainingData['Direction']
validationData = weekly[weekly.Year > 2008]
X_vd = validationData['Lag2'].reshape(-1,1)
y_vd = validationData['Direction']



# In[93]:

# D) Continued.
model_LR = skl_lm.LogisticRegression()
model_LR.fit(X_td, y_td)
predicted_LR = model_LR.predict(X_vd)
cm_LR = confusion_matrix(y_vd, predicted_LR, labels=["Up","Down"])
print("Confusion Matrix: \n" + str(cm_LR))
print("Fraction Correct: " + str(cm_LR[0,0] + cm_LR[1,1]) + "/" + str(cm_LR.sum()))
print("Percent Correct: " + str(((cm_LR[0,0] + cm_LR[1,1]) / float(cm_LR.sum()))*100))


# In[94]:

# E) Repeat (d) using LDA.
model_LDA = LinearDiscriminantAnalysis()
model_LDA.fit(X_td,y_td)
predicted_LDA = model_LDA.predict(X_vd)
cm_LDA = confusion_matrix(y_vd, predicted_LDA, labels = ["Up", "Down"])
print("Confusion Matrix: \n" + str(cm_LDA))
print("Fraction Correct: " + str(cm_LDA[0,0] + cm_LDA[1,1]) + "/" + str(cm_LDA.sum()))
print("Percent Correct: " + str(((cm_LDA[0,0] + cm_LDA[1,1]) / float(cm_LDA.sum()))*100))


# In[95]:

# F) Repeat (d) using QDA.
model_QDA = QuadraticDiscriminantAnalysis()
model_QDA.fit(X_td,y_td)
predicted_QDA = model_QDA.predict(X_vd)
cm_QDA = confusion_matrix(y_vd, predicted_QDA, labels = ["Up", "Down"])
print("Confusion Matrix: \n" + str(cm_QDA))
print("Fraction Correct: " + str(cm_QDA[0,0] + cm_QDA[1,1]) + "/" + str(cm_QDA.sum()))
print("Percent Correct: " + str(((cm_QDA[0,0] + cm_QDA[1,1]) / float(cm_QDA.sum()))*100))


# In[97]:

# G) Repeat (d) using KNN with K = 1.
k = 1
model_KNN = KNeighborsClassifier(n_neighbors = k)
model_KNN = model_KNN.fit(X_td, y_td)
predicted_KNN = model_KNN.predict(X_vd)
cm_KNN = confusion_matrix(y_vd, predicted_KNN, labels = ["Up","Down"])
print("Confusion Matrix: \n" + str(cm_KNN))
print("Fraction Correct: " + str(cm_KNN[0,0] + cm_KNN[1,1]) + "/" + str(cm_KNN.sum()))
print("Percent Correct: " + str(((cm_KNN[0,0] + cm_KNN[1,1]) / float(cm_KNN.sum()))*100))


# In[ ]:

# H) Which of these methods appears to provide the best results on
# this data?


# In[ ]:

# I) Experiment with different combinations of predictors, including 
# possible transformations and interactions, for each of the methods. 
# Report the variables, method, and associated confusion matrix that 
# appears to provide the best results on the held out data. Note that 
# you should also experiment with values for K in the KNN classifier.


# In[ ]:




# In[98]:

# Chapter 5 Problem 5
# Estimate the test error of this logistic regression model using the 
# validation set approach. Do not forget to set a random seed before 
# beginning your analysis
default = pd.read_csv('./input/Default.csv')
default = default.iloc[:,1:]
default.info()
print(default.head())
print(default.corr())
scatter_matrix(default)
plt.show()


# In[102]:

# Digging through the data to get some insight
default_Yes = default.loc[default['default'] == 'Yes']
default_No = default.loc[default['default'] == 'No']
plt.scatter(default_No['income'], default_No['balance'], c='b')
plt.scatter(default_Yes['income'], default_Yes['balance'], c='r')
maxVal = -1
minBal = 0
for balance in range(1000, 2750):
    count_All = default.loc[(default['balance'] > balance)]
    count_default = default.loc[(default['default'] == 'Yes')&(default['balance'] > balance)]
    if(len(count_All) == 0):
        continue
    if len(count_default)/float(len(count_All)) > maxVal:
        maxVal = len(count_default)/float(len(count_All))
        minBal = balance
print("Percent of people defaulting maximizes at %s at a balance of %s" %(maxVal, minBal))
plt.show()

default_Student = default.loc[default['student'] == 'Yes']
default_Yes = default_Student.loc[default_Student['default'] == 'Yes']
default_No = default_Student.loc[default_Student['default'] == 'No']
plt.scatter(default_No['income'], default_No['balance'], c='b')
plt.scatter(default_Yes['income'], default_Yes['balance'], c='r')

maxVal = -1
minBal = 0
for balance in range(1000, 2750):
    count_All = default_Student.loc[(default_Student['balance'] > balance)]
    count_default = default_Student.loc[(default_Student['default'] == 'Yes')&(default_Student['balance'] > balance)]
    if(len(count_All) == 0):
        continue
    if len(count_default)/float(len(count_All)) > maxVal:
        maxVal = len(count_default)/float(len(count_All))
        minBal = balance
print("Percent of Students defaulting maximizes at %s at a balance of %s" %(maxVal, minBal))
plt.show()


# In[100]:

# A) Fit a logistic regression model that uses
# income and balance to predict default.
model_LR1 = skl_lm.LogisticRegression()
model_LR1 = model_LR.fit(default.iloc[:,2:], default.iloc[:,0])


# In[107]:

# B) Using the validation set approach, estimate the test error of this
# model. In order to do this, you must perform the following steps:

# predictors are in the 1-n columns, result is in column 0
def classify(data, trainingFraction):
    trainingBound = int(len(data)*trainingFraction)
    print("----------START----------")
    print("%s values in training set." %(trainingBound))
    data = data.sample(frac=1)
    # i. Split the sample set into a training set and a validation set.
    X_td = data.iloc[:trainingBound,1:]
    X_vd = data.iloc[trainingBound:,1:]
    y_td = data.iloc[:trainingBound,0]
    y_vd = data.iloc[trainingBound:,0]

    # ii. Fit a multiple logistic regression model using only the training
    # observations.
    model_LR = skl_lm.LogisticRegression()
    model_LR = model_LR.fit(X_td, y_td)
    
    # iii. Obtain a prediction of default status for each individual in the 
    # validation set by computing the posterior probability of default for 
    # that individual, and classifying the individual to the default category 
    # if the posterior probability is greater than 0.5.
    predictions = model_LR.predict(X_vd)
          
    # iv. Compute the validation set error, which is the fraction of the 
    # observations in the validation set that are misclassified.
    vse = model_LR.score(X_vd, y_vd)
    return(vse)

data = default.drop('student',1 )
vse = classify(data, .90)
print("Validation Set Error: %s" %(vse))
    


# In[110]:

# C) Repeat the process in (b) three times, using three different 
# splits of the observations into a training set and a validation set. 
# Comment on the results obtained.
data = default.drop('student',1 )
total1 = sum(classify(data,0.1) for i in range(10))
total2 = sum(classify(data,0.5) for i in range(10))
total3 = sum(classify(data,0.9) for i in range(10))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total1/10.))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total2/10.))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total3/10.))


# In[109]:

# D) Now consider a logistic regression model that predicts the 
# probability of default using income, balance, and a dummy variable 
# for student. Estimate the test error for this model using the 
# validation set approach. Comment on whether or not including a 
# dummy variable for student leads to a reduction in the test error rate.
data = default.drop('student',1)
data['student_dummy'] = default.student == 'Yes'
total1 = sum(classify(data,0.1) for i in range(10))
total2 = sum(classify(data,0.5) for i in range(10))
total3 = sum(classify(data,0.9) for i in range(10))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total1/10.))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total2/10.))
print("\n***AVERAGE*** \n Validation Set Error: %s\n*************" %(total3/10.))


# coding: utf-8

# In[317]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Number 6 Chapter 5 Problem 8

# In[318]:

from sklearn.linear_model import LogisticRegression
np.random.seed(42)
y = np.random.normal(size = 100)
#print(np.shape(y))
x = np.random.normal(size = 100)
#print(x)
y = x - 2*x**2 + np.random.normal(size = 100)
print(np.shape(y))
plt.scatter(x,y)
plt.show()
print("The plot shows a curve opening down")


# In[319]:

def func1(x, b0, b1, c):
    return b0 + b1*x + c
def func2(x, b0, b1, b2, c):
    return b0 + b1*x + b2*x**2 + c
def func3(x, b0, b1, b2, b3, c):
    return b0 + b1*x + b2*x**2 + b3*x**3 + c
def func4(x, b0, b1, b2, b3, b4, c):
    return b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 +c


# In[320]:

import scipy.optimize as optimization
z = np.zeros(100)
popt1, pcov1 = optimization.curve_fit(func1, x, y)
popt2, pcov2 = optimization.curve_fit(func2, x, y)
popt3, pcov3 = optimization.curve_fit(func3, x, y)
popt4, pcov4 = optimization.curve_fit(func4, x, y)
print(popt1, popt2, popt3, popt4)


# In[321]:

y1 = func1(x, popt1[0], popt1[1], popt1[2])
plt.scatter(x,y)
plt.scatter(x,y1,color='red')
plt.show()


# In[322]:

y2 = popt2[0] + popt2[1]*x + popt2[2]*x**2 + popt2[3]
plt.scatter(x,y)
plt.scatter(x,y2,color='red')
plt.show()


# In[323]:

y3 = popt3[0] + popt3[1]*x + popt3[2]*x**2 + popt3[3]*x**3 + popt3[4]
plt.scatter(x,y)
plt.scatter(x,y3,color='red')
plt.show()


# In[324]:

y4 = popt4[0] + popt4[1]*x + popt4[2]*x**2 + popt4[3]*x**3 + popt4[4]*x**4 + popt4[5]
plt.scatter(x,y)
plt.scatter(x,y4,color='red')
plt.show()


# In[325]:

from sklearn import model_selection
loo = model_selection.LeaveOneOut()
np.random.seed(1)
diff1 = 0
for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    weights, throw_away = optimization.curve_fit(func1, x_train, y_train)
    diff1 += (y_test - func1(x_test, weights[0], weights[1], weights[2]))**2
    #print(y_test - func1(x_test, popt1[0], popt1[1], popt1[2]))
diff1 /= 100
print(diff1)
    #print(x_train, x_test


# In[326]:

diff2 = 0
for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    weights, throw_away = optimization.curve_fit(func2, x_train, y_train)
    diff2 += (y_test - func2(x_test, weights[0], weights[1], weights[2], weights[3]))**2
    
diff2 /= 100
print(diff2)


# In[327]:

diff3 = 0
for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    weights, throw_away = optimization.curve_fit(func3, x_train, y_train)
    diff3 += (y_test - func3(x_test, weights[0], weights[1], weights[2], weights[3], weights[4]))**2
    
diff3 /= 100
print(diff3)


# In[328]:

diff4 = 0
for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    weights, throw_away = optimization.curve_fit(func4, x_train, y_train)
    diff4 += (y_test - func4(x_test, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]))**2
    
diff4 /= 100
print(diff4)


# # Problem 6: Problem 9 from Chapter 6

# In[342]:

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import cross_validation


# In[333]:

college = pd.read_csv("../Data/Data/College.csv")
college = college.replace(['Yes', 'No'], [1,0])
college.shape


# In[334]:

#Part A
features = ['Private', 'Accept', 'Enroll', 'Top10perc', 'F.Undergrad',
            'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal',
            'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']
X_train, X_test, Y_train, Y_test = train_test_split(college[features], college['Apps'], test_size=0.4, random_state=42)


# In[335]:

#Part B
lms = linear_model.LinearRegression()
lms.fit(X_train, Y_train)

score = mean_squared_error(Y_test, lms.predict(X_test))
print("Error:", score)


# In[336]:

#Part C
ridge_reg = linear_model.RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10])
ridge_reg.fit(X_train, Y_train)

score = mean_squared_error(Y_test, ridge_reg.predict(X_test))
print("Error:", score)


# In[337]:

#Part D
lasso_reg = linear_model.LassoCV(alphas=[0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10])
lasso_reg.fit(X_train, Y_train)

score = mean_squared_error(Y_test, lasso_reg.predict(X_test))
print("Error:", score)


# In[347]:

pca = PCA()
X_reduced = pca.fit_transform(X_train)

n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

regr = linear_model.LinearRegression()
mse = []

score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)

for i in np.arange(1,11):
    score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(mse, '-v')
ax2.plot([1,2,3,4,5,6,7,8,9,10], mse[1:11], '-v')
ax2.set_title('Intercept excluded from plot')

for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    ax.set_xlim((-0.2,10.2))
    
plt.show()


# In[349]:

pcr_regr = linear_model.LinearRegression()
pcr_regr.fit(X_reduced[:,:5], Y_train)

score = mean_squared_error(Y_test, ridge_reg.predict(X_test))
print('Error: {}', score)
print('Value of M selected by CV: 5')


# In[339]:

#Part F
from sklearn.cross_decomposition import PLSRegression
params = {'n_components':[2, 3, 4, 5, 7, 10]}

pls = PLSRegression()
pls_reg = GridSearchCV(pls, params, scoring='neg_mean_squared_error')
pls_reg.fit(X_train, Y_train)

score = mean_squared_error(Y_test, pls_reg.predict(X_test))
print("Error:", score)
print("Value of M selected by CV: {}".format(pls_reg.best_params_['n_components']))


# # Problem 8 Chapter 6 Problem 11

# In[382]:

boston = pd.read_csv('../Data/Data/Boston.csv')
list(boston)


# In[390]:

features = ['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
X_train, X_test, Y_train, Y_test = train_test_split(boston[features], boston['crim'], test_size=0.4, random_state=42)


# In[391]:

lms = linear_model.LinearRegression()
lms.fit(X_train, Y_train)
print(lms.coef_)

#score = mean_squared_error(Y_test, lms.predict(X_test))
score = lms.score(X_test, Y_test)

print("Error:", score)


# In[392]:

ridge_reg = linear_model.RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10])
ridge_reg.fit(X_train, Y_train)

#score = mean_squared_error(Y_test, ridge_reg.predict(X_test))
print(ridge_reg.coef_)
score = ridge_reg.score(X_test, Y_test)
print("Error:", score)


# In[393]:

lasso_reg = linear_model.LassoCV(alphas=[0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10])
lasso_reg.fit(X_train, Y_train)

#score = mean_squared_error(Y_test, lasso_reg.predict(X_test))
score = lasso_reg.score(X_test, Y_test)
print("Error:", score)


# ## Ridge Regression Model fits the best for predicting crime rates in Boston we used all the features in our prediction


# Problem 9
# In this problem we will use synthetic data sets to explore the 
# bias-variance tradeoff incurred by using regularization.
def syntheticData():
    n = 51
    p = 50
    b = [1]*p
    X = np.random.normal(0,1, size=(n,p))
    noise = np.random.normal(0,.25,n)
    y = X.dot(b) + noise
    return(X,y)


# In[126]:

# A) Estimate the mean and variance of βhat, for only a single component. 
# we will choose to do the mean and var for b0
num = 1000
b_hat = np.array([None] * num)
for i in range(num):
    X,y = syntheticData()
    model_LSR = skl_lm.LinearRegression()
    model_LSR = model_LSR.fit(X,y)
    b_hat[i] = model_LSR.coef_[0]
print("Mean:\t%s" %(b_hat.mean()))
print("Var:\t%s" %(b_hat.var()))
# This result still varies significantly. That makes sens though...To an extent


# In[128]:

# B) Choose regularization coefficients λ = 0.01,0.1,1,10,100 and repeat the above experiment.
alphas = [0.01, 0.1, 1, 10, 100]
num = 1000
for alpha in alphas:
    b_hat = np.array([None] * num)
    for i in range(num):
        X,y = syntheticData()
        model_RR = skl_lm.Ridge(alpha)
        model_RR = model_RR.fit(X,y)
        b_hat[i] = model_RR.coef_[0]
    print("Lambda = %s" %(alpha))
    print("Mean:\t%s" %(b_hat.mean()))
    print("Var:\t%s" %(b_hat.var()))
    print("---------------------------")


# In[4]:

# Problem 8: Chapter 6 Problem 11
boston = pd.read_csv('input/Boston.csv')
boston.info()
boston.head()
print boston.corr()


# In[194]:

# A) Try out some of the regression methods explored in this chapter, 
# such as best subset selection, the lasso, ridge regression, and PCR. 
# Present and discuss results for the approaches that you consider.
# Subset Selection
alphas = [0.001,0.01,0.1,1,10,100]
# Lasso
for alpha in alphas:
    model_L = skl_lm.LassoCV(alpha)
    model_L = model_L.fit(boston.iloc[:,2:], boston.iloc[:,1])
    score = model_L.score(boston.iloc[:,2:], boston.iloc[:,1])
    print "Alpha: %s\tScore: %s" %(alpha, score)
# Ridge
for alpha in alphas:
    model_RR = skl_lm.Ridge(alpha)
    model_RR =model_RR.fit(boston.iloc[:,2:], boston.iloc[:,1])
    score = model_RR.score(boston.iloc[:,2:], boston.iloc[:,1])
    print "Alpha: %s\tScore: %s" %(alpha, score)
# PCR


# In[229]:

# B) Propose a model (or set of models) that seem to perform well on this 
# data set, and justify your answer. Make sure that you are evaluating 
# model performance using validation set error, cross- validation, or some
# other reasonable alternative, as opposed to using training error.
trainingFraction = 0.9

trainingBound = int(round(len(boston) * trainingFraction))
for i in range(2,15):
    for j in range(i+1,15):
        X_td = boston.iloc[:trainingBound,i:j]
        X_vd = boston.iloc[trainingBound:,i:j]
        y_td = boston.iloc[:trainingBound,1]
        y_vd = boston.iloc[trainingBound:,1]

        model_L = skl_lm.Ridge(.01)
        model_L = model_L.fit(X_td, y_td)
        print model_L.score(X_vd, y_vd)