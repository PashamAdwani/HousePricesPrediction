import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import time
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

    
input_cols=['Id', 'MSSubClass', 'MSZoning_NUM', 'LotFrontage',
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
    #Read
    df_test=pd.read_csv(r'D:\Data Analysis\Sparsh\P2\Test.csv',header=0,sep=',')
    df_train=pd.read_csv(r'D:\Data Analysis\Sparsh\P2\Train.csv',header=0,sep=',')
    cols=df_train.columns
    #inputCols=df_train.columns[0:len(cols)-1]
    y_train=df_train['SalePrice']
    dt = DecisionTreeClassifier(random_state=99)
    dt.fit(df_train[input_cols], df_train['SalePrice']) # fitting the data in the model 
    ypred=dt.predict(df_test[input_cols])
    #Tolerence
    steps=[]
    i=0
    while i<=1:
        steps.append(i)
        i=i+0.05
    count=0
    score=np.zeros((len(steps),1))
    less=[]
    more=[]
    pred=[]
    SP=[]         #sale_price
    x_axis=[]
    c=1
    for j in steps:
        for i in range(len(ypred)):
            SP_less=df_test['SalePrice'][i]*(1-j)
            SP_more=df_test['SalePrice'][i]*(1+j)
            if(j==0.49999999999999994):
                less.append(SP_less)
                more.append(SP_more)
                pred.append(ypred[i])
                SP.append(df_test['SalePrice'][i])
                x_axis.append(c)
                c=c+1
            if(ypred[i]>=SP_less and ypred[i]<=SP_more):
                score[count]=score[count]+1
        count=count+1
    count=0
    a=[]
    for i in range(len(score)):
        x=score[i]*100/len(ypred)
        a.append(x)
        print('For tolerence ',steps[i], ' accuracy is ',x)
    #visualize_tree(dt, input_cols)
    #plot
    plt.plot(x_axis[0:10], less[0:10], 'ro', x_axis[0:10], more[0:10], 'bo', pred[0:10], 'go',x_axis[0:10],SP[0:10],'ko')
    plt.show()
    plt.plot(steps,a,'r--')
    plt.show()
else:
    print('The Files/path DO NOT exist. Either change the path or run the data file first if you havent already done so.')
    
    
