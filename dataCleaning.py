import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def change_to_num(oldColName,newColName,df):
    x=df[oldColName].unique()
    CTN={name:num for num,name in enumerate(x)}
    df[newColName]=df[oldColName].replace(CTN)
    return (df[newColName])


df_main=pd.read_csv(r'D:\Data Analysis\Sparsh\P2\data\data\data.csv',header=0,sep=',')

#Data Cleaning
#lenUniqueCol=np.zeros((len(df_main.columns)))
nullCol=[]
for i in range(len(df_main.columns)):
    z=df_main.columns[i]
    if(df_main[z].isna().any()==False):
        print(z,' does not have null/empty columns')
    else:
        print(z,' has null values')
        nullCol.append(z)
        
#Filling the null/empty values
for i in nullCol:
	print(i, ' is of type: ',type(df_main[i][0]).__name__)
            
df_main['LotFrontage']=df_main['LotFrontage'].fillna(0)            
df_main['Alley']=df_main['Alley'].fillna('No Alley')
df_main['MasVnrType']=df_main['MasVnrType'].fillna('None')
df_main['MasVnrArea']=df_main['MasVnrArea'].fillna(0)
df_main['BsmtQual']=df_main['BsmtQual'].fillna('None')
df_main['BsmtCond']=df_main['BsmtCond'].fillna('None')
df_main['BsmtExposure']=df_main['BsmtExposure'].fillna('No')
df_main['BsmtFinType1']=df_main['BsmtFinType1'].fillna('No Basement')
df_main['BsmtFinType2']=df_main['BsmtFinType2'].fillna('No Basement')
df_main['Electrical']=df_main['Electrical'].fillna('None')
df_main['FireplaceQu']=df_main['FireplaceQu'].fillna('No FP')
df_main['GarageType']=df_main['GarageType'].fillna('None')
df_main['GarageFinish']=df_main['GarageFinish'].fillna('None')
df_main['GarageYrBlt']=df_main['GarageYrBlt'].fillna(df_main['GarageYrBlt'].mode())
df_main['GarageQual']=df_main['GarageQual'].fillna('None')
df_main['GarageCond']=df_main['GarageCond'].fillna('None')
df_main['PoolQC']=df_main['PoolQC'].fillna('No Pool')
df_main['Fence']=df_main['Fence'].fillna('No Fence')

df_copy=df_main
finalCols=[]
for i in df_copy.columns:
    oldColName=i
    if(type(df_copy[i][5]).__name__=='str'):
        newColName=oldColName+'_NUM'
        df_copy[newColName]=change_to_num(oldColName,newColName,df_copy)
        finalCols.append(newColName)
    else:
        finalCols.append(i)

#Splitting into test train split

train,test,y_train,y_test=train_test_split(df_copy[finalCols[0:80]],df_copy['SalePrice'],test_size=0.2, random_state=77)


#Writing the reaining and testing data to csv
#Train
train['SalePrice']=y_train
train.to_csv(r'D:\Data Analysis\Sparsh\P2\Train.csv', sep=',')
#Test
test['SalePrice']=y_test
test.to_csv(r'D:\Data Analysis\Sparsh\P2\Test.csv', sep=',')
        
