import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import time
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
from sklearn.model_selection import train_test_split


def change_to_num(oldColName, newColName, df):
    x = df[oldColName].unique()
    CTN = {name: num for num, name in enumerate(x)}
    df[newColName] = df[oldColName].replace(CTN)
    return (df[newColName])


df_main = pd.read_csv("/Users/sparshgoyal/Downloads/data/data.csv", header=0, sep=',')

# Data Cleaning
# lenUniqueCol=np.zeros((len(df_main.columns)))
nullCol = []
for i in range(len(df_main.columns)):
    z = df_main.columns[i]
    if (df_main[z].isna().any() == False):
        print(z, ' does not have null/empty columns')
    else:
        print(z, ' has null values')
        nullCol.append(z)

# Filling the null/empty values
for i in nullCol:
    print(i, ' is of type: ', type(df_main[i][0]).__name__)

df_main['LotFrontage'] = df_main['LotFrontage'].fillna(0)
df_main['Alley'] = df_main['Alley'].fillna('No Alley')
df_main['MasVnrType'] = df_main['MasVnrType'].fillna('None')
df_main['MasVnrArea'] = df_main['MasVnrArea'].fillna(0)
df_main['BsmtQual'] = df_main['BsmtQual'].fillna('None')
df_main['BsmtCond'] = df_main['BsmtCond'].fillna('None')
df_main['BsmtExposure'] = df_main['BsmtExposure'].fillna('No')
df_main['BsmtFinType1'] = df_main['BsmtFinType1'].fillna('No Basement')
df_main['BsmtFinType2'] = df_main['BsmtFinType2'].fillna('No Basement')
df_main['Electrical'] = df_main['Electrical'].fillna('None')
df_main['FireplaceQu'] = df_main['FireplaceQu'].fillna('No FP')
df_main['GarageType'] = df_main['GarageType'].fillna('None')
df_main['GarageFinish'] = df_main['GarageFinish'].fillna('None')
df_main['GarageYrBlt'] = df_main['GarageYrBlt'].fillna(df_main['GarageYrBlt'].mode())
df_main['GarageQual'] = df_main['GarageQual'].fillna('None')
df_main['GarageCond'] = df_main['GarageCond'].fillna('None')
df_main['PoolQC'] = df_main['PoolQC'].fillna('No Pool')
df_main['Fence'] = df_main['Fence'].fillna('No Fence')

df_copy = df_main
finalCols = []
for i in df_copy.columns:
    oldColName = i
    if (type(df_copy[i][5]).__name__ == 'str'):
        newColName = oldColName + '_NUM'
        df_copy[newColName] = change_to_num(oldColName, newColName, df_copy)
        finalCols.append(newColName)
    else:
        finalCols.append(i)

# Splitting into test train split

train, test, y_train, y_test = train_test_split(df_copy[finalCols[0:80]], df_copy['SalePrice'], test_size=0.2,
                                                random_state=77)

# Writing the reaining and testing data to csv
# Train
train['SalePrice'] = y_train
train.to_csv("/Users/sparshgoyal/Downloads/data/train.csv", sep=',')
# Test
test['SalePrice'] = y_test
test.to_csv("/Users/sparshgoyal/Downloads/data/test.csv", sep=',')






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
if os.path.exists("/Users/sparshgoyal/Downloads/data/data.csv"):
    #Real
    df_test=pd.read_csv("/Users/sparshgoyal/Downloads/data/train.csv",header=0,sep=',')
    df_train=pd.read_csv("/Users/sparshgoyal/Downloads/data/test.csv",header=0,sep=',')
    cols=df_train.columns
    inputCols=df_train.columns[0:len(cols)-1]
    y_train=df_train['SalePrice']
    reg = LinearRegression().fit(df_train[input_cols], df_train['SalePrice'])
    # fitting the data in the model 
    ypred=reg.predict(df_test[input_cols])
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
    SP=[]
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
    print("----------------------------------------------------------------------------------")
    print("--------------Accuracy in Linear Regression Model using Tolerance-----------------")
    for i in range(len(score)):
        x=score[i]*100/len(ypred)
        a.append(x)
        print('For tolerence ',steps[i], ' accuracy is ',x)
    #visualize_tree(dt, input_cols)
    #plot
    plt.plot(steps,a,'r--')
    plt.title("Linear Regression Model")
    plt.ylabel("Number of correct predictions")
    plt.xlabel("Tolerance")
    plt.show()
else:
    print('The Files/path DO NOT exist. Either change the path or run the data file first if you havent already done so.')


#------------------------------------------For Decision Tree-------------------------------------------

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


input_cols = ['Id', 'MSSubClass', 'MSZoning_NUM', 'LotFrontage',
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

# Read csv files
if os.path.exists("/Users/sparshgoyal/Downloads/data/data.csv"):
    # Read
    df_test = pd.read_csv("/Users/sparshgoyal/Downloads/data/test.csv", header=0, sep=',')
    df_train = pd.read_csv("/Users/sparshgoyal/Downloads/data/train.csv", header=0, sep=',')
    cols = df_train.columns
    # inputCols=df_train.columns[0:len(cols)-1]
    y_train = df_train['SalePrice']
    dt = DecisionTreeClassifier(random_state=99)
    dt.fit(df_train[input_cols], df_train['SalePrice'])  # fitting the data in the model
    ypred = dt.predict(df_test[input_cols])
    # Tolerence
    step = []
    i = 0
    while i <= 1:
        step.append(i)
        i = i + 0.05
    count = 0
    score = np.zeros((len(step), 1))
    less = []
    more = []
    pred = []
    SP = []  # sale_price
    x_axis = []
    c = 1
    for j in step:
        for i in range(len(ypred)):
            SP_less = df_test['SalePrice'][i] * (1 - j)
            SP_more = df_test['SalePrice'][i] * (1 + j)
            if (j == 0.49999999999999994):
                less.append(SP_less)
                more.append(SP_more)
                pred.append(ypred[i])
                SP.append(df_test['SalePrice'][i])
                x_axis.append(c)
                c = c + 1
            if (ypred[i] >= SP_less and ypred[i] <= SP_more):
                score[count] = score[count] + 1
        count = count + 1
    count = 0
    b = []
    print("----------------------------------------------------------------------------------")
    print("-----------------Accuracy in Decision tree Model using Tolerance------------------")
    for i in range(len(score)):
        x = score[i] * 100 / len(ypred)
        b.append(x)
        print('For tolerence ', step[i], ' accuracy is ', x)
    # visualize_tree(dt, input_cols)
    # plot
    plt.plot(step, b, 'r--')
    plt.title("Decision Tree Model")
    plt.ylabel("Number of Correct Predictions")
    plt.xlabel("Tolerance")
    plt.show()
else:
    print(
        'The Files/path DO NOT exist. Either change the path or run the data file first if you havent already done so.')
print("----------------------------------------------------------------------------------")



