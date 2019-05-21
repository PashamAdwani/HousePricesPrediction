import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import time
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize

    
input_cols=['MSSubClass', 'MSZoning_NUM', 'LotFrontage',
       'LotArea', 'Street_NUM', 'Alley_NUM', 'LotShape_NUM', 'LandContour_NUM',
       'Utilities_NUM', 'LotConfig_NUM', 'LandSlope_NUM', 'Neighborhood_NUM',
       'Condition1_NUM', 'Condition2_NUM', 'BldgType_NUM', 'HouseStyle_NUM',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle_NUM', 'RoofMatl_NUM', 'Exterior1st_NUM', 'Exterior2nd_NUM',
       'MasVnrType_NUM', 'MasVnrArea', 'ExterQual_NUM', 'ExterCond_NUM',
       'Foundation_NUM', 'BsmtQual_NUM', 'BsmtCond_NUM', 'BsmtExposure_NUM',
       'BsmtFinType1_NUM', 'BsmtFinSF1', 'BsmtFinType2_NUM', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating_NUM', 'HeatingQC_NUM',
       'CentralAir_NUM', 'Electrical_NUM', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual_NUM',
       'TotRmsAbvGrd', 'Functional_NUM', 'Fireplaces', 'FireplaceQu_NUM',
       'GarageType_NUM', 'GarageFinish_NUM', 'GarageCars',
       'GarageArea', 'GarageQual_NUM', 'GarageCond_NUM', 'PavedDrive_NUM',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC_NUM', 'Fence_NUM', 'YrSold', 'SaleType_NUM', 'SaleCondition_NUM']    

#Read csv files
if os.path.exists(r'D:\Data Analysis\Sparsh\P2\Test.csv'):
    #Real
    df_test=pd.read_csv(r'D:\Data Analysis\Sparsh\P2\Test.csv',header=0,sep=',')
    df_train=pd.read_csv(r'D:\Data Analysis\Sparsh\P2\Train.csv',header=0,sep=',')
    cols=df_train.columns
    inputCols=df_train.columns[0:len(cols)-1]
    y_train=df_train['SalePrice']
    # fitting the data in the LINEAR REGRESS MODEL
    reg = LinearRegression().fit(df_train[input_cols], df_train['SalePrice'])
    regypred=reg.predict(df_test[input_cols])
    #Fitting Model in DECISION TREE CLASSIFIER
    dt = DecisionTreeClassifier(random_state=99)
    dt.fit(df_train[input_cols], df_train['SalePrice'])
    dtypred=dt.predict(df_test[input_cols])
    #Tolerence
    steps=[]
    i=0
    while i<=1:
        steps.append(i)
        i=i+0.05
    count=0
    regscore=np.zeros((len(steps),1))
    dtscore=np.zeros((len(steps),1))
    less=[]
    more=[]
    dtpred=[]
    regpred=[]
    SP=[]
    x_axis=[]
    c=1
    dtsum=0
    regsum=0
    for j in steps:
        for i in range(len(dtypred)):
            dtscore=dtscore+(abs(dtypred[i]-df_test['SalePrice'][i]))
            regscore=regscore+(abs(regypred[i]-df_test['SalePrice'][i]))
            SP_less=df_test['SalePrice'][i]*(1-j)
            SP_more=df_test['SalePrice'][i]*(1+j)
            if(regypred[i]>=SP_less and regypred[i]<=SP_more):
                regscore[count]=regscore[count]+1
            if(dtypred[i]>=SP_less and dtypred[i]<=SP_more):
                dtscore[count]=dtscore[count]+1    
        count=count+1
    count=0
    dta=[]
    rega=[]
    for i in range(len(dtscore)):
        x=dtscore[i]*100/len(dtypred)
        dta.append(x)
        print('For tolerence ',steps[i], ' accuracy is ',x)
    for i in range(len(regscore)):
        x=regscore[i]*100/len(regypred)
        rega.append(x)
        print('For tolerence ',steps[i], ' accuracy is ',x)    
    #visualize_tree(dt, input_cols)
    #plot
    plt.plot(steps,rega,'r--', steps,dta,'b--')#,label='Decision Tree')
    plt.xlabel('Tolerance')
    plt.ylabel('Accuracy')
    plt.title('Linear Regression & Decision Tree Accuracy by Tolerance')
    plt.show()
else:
    print('The Files/path DO NOT exist. Either change the path or run the data file first if you havent already done so.')
    
