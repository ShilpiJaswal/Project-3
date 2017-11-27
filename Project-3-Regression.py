
# coding: utf-8

# In[6]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer
from sklearn.model_selection import cross_val_score, train_test_split, KFold 
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, BayesianRidge
from sklearn.metrics import mean_squared_error, make_scorer
# from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


get_ipython().magic('matplotlib inline')


# In[7]:

#Reading in our dataset
train_df = pd.read_csv('train.csv') # reading in the training dataset.
test_df = pd.read_csv('test.csv')# reading in the test dataset.
combined = pd.concat([train_df,test_df], axis=0) #combining train and test dataset for data preprocessing
combined.shape


# In[8]:

train_df.shape # training dataset has 1460 observationsb


# In[9]:

test_df.info()


# In[10]:

test_df.shape # test dataset has 1459 observations


# In[11]:

combined.head()


# In[12]:

#How many columns with different data types are there?
combined.get_dtype_counts()


# In[13]:

combined['SalePrice'].describe() # checking how Sales Price Variable is distributed



# In[14]:

#plot the distribution plot of SalePrices of the houses
plt.figure(figsize=(12,6))
sns.distplot(combined['SalePrice'].dropna() ,kde= False,bins=75 , rug = True ,color='purple')
sns.set(font_scale = 1.25)
plt.tight_layout()
plt.title('Distribution of Sale Price')


# In[15]:

sp_corr = combined.corr()["SalePrice"]
sp_corr_sort = sp_corr.sort_values(axis = 0 , ascending = False)
sp_corr_sort[sp_corr_sort > 0.50]


# In[16]:

combined.columns = [x.replace(' ','') for x in combined.columns]


# In[17]:

combined.columns


# In[18]:

corr = combined[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemod/Add"]].corr()


# In[19]:

plt.figure(figsize=(18,10))
sns.heatmap(corr, linecolor= "white" , lw =1,annot=True)


# In[20]:

sns.boxplot(y="SalePrice", x="OverallQual", data=combined)
plt.title('Overall Quality vs SalePrice')


# In[21]:

sns.jointplot(y="SalePrice", x="GrLivArea", data=combined)
plt.title('Ground Living Area vs SalePrice')


# In[22]:

sns.jointplot(y="SalePrice", x="GarageArea", data=combined)
plt.title('GarageArea vs SalePrice')


# In[23]:

sns.jointplot(y="SalePrice", x="1stFlrSF", data=combined)
plt.title('1st Floor Surface Area vs SalePrice')


# In[24]:

sns.boxplot(y="SalePrice", x="FullBath", data=combined)
plt.title('FullBath Area vs SalePrice')


# In[25]:

sns.jointplot(y="SalePrice", x="YearBuilt", data=combined)
plt.title('YearBuilt vs SalePrice')


# In[26]:

#Missing Value imputation
#Categorical variables
combined.select_dtypes(include=['object']).columns


# In[27]:

#Numerical Columns
combined.select_dtypes(include=['float64', 'int64']).columns



# In[28]:

mean_imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
median_imputer = Imputer(missing_values='NaN', strategy = 'median', axis=0)
mode_imputer = Imputer(missing_values='NaN', strategy = 'most_frequent', axis=0)


# In[29]:

#Missing values in the columns
combined[combined.columns[combined.isnull().any()]].isnull().sum()


# In[30]:

sns.countplot(x = 'Alley' , data = combined )


# In[31]:

combined['Alley'].fillna('None',inplace = True)


# In[32]:

combined[combined['BsmtCond'].isnull() == True][['BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'BsmtQual','BsmtFinSF1',
       'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF','TotalBsmtSF']]


# In[33]:

#Categorical features 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'BsmtQual'

combined['BsmtQual'].fillna(value = 'None' , inplace = True)
combined['BsmtCond'].fillna(value = 'None' , inplace = True)
combined['BsmtExposure'].fillna(value = 'None' , inplace = True)
combined['BsmtFinType1'].fillna(value = 'None' , inplace = True)
combined['BsmtFinType2'].fillna(value = 'None' , inplace = True)


# In[34]:

#Numerical Features 'BsmtCond','BsmtFinSF1','BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF','TotalBsmtSF'

combined['BsmtFinSF1'].fillna(value = 0 , inplace = True)
combined['BsmtFinSF2'].fillna(value = 0 , inplace = True)
combined['BsmtFullBath'].fillna(value = 0 , inplace = True)
combined['BsmtHalfBath'].fillna(value = 0 , inplace = True)
combined['BsmtUnfSF'].fillna(value = 0 , inplace = True)
combined['TotalBsmtSF'].fillna(value = 0 , inplace = True)


# In[35]:

sns.countplot(x = 'Electrical' , data = combined)



# In[36]:

combined['Electrical'].fillna(value = 'SBrkr' , inplace = True)


# In[37]:

combined[combined['FireplaceQu'].isnull() == True][['Fireplaces','FireplaceQu']]



# In[38]:

combined['FireplaceQu'].fillna(value = 'None' , inplace =  True)


# In[39]:

combined[combined['GarageType'].isnull() == True][['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageArea','GarageCars']]



# In[40]:

combined['GarageType'].fillna(value = 'None' , inplace = True)
combined['GarageYrBlt'].fillna(value = 'None' , inplace = True)
combined['GarageFinish'].fillna(value = 'None' , inplace = True)
combined['GarageQual'].fillna(value = 'None' , inplace = True)
combined['GarageCond'].fillna(value = 'None' , inplace = True)
combined['GarageArea'].fillna(value = 0 , inplace = True)
combined['GarageCars'].fillna(value = 0 , inplace = True)


# In[41]:

sns.countplot(x = 'PoolQC' , data = combined)


# In[42]:

combined[combined['PoolQC'].isnull() == True][['PoolQC','PoolArea']]


# In[43]:

combined['PoolQC'].fillna(value = 'None' , inplace = True)


# In[44]:

sns.distplot(combined['LotFrontage'].dropna() , bins =70)


# In[45]:

combined['LotFrontage'] = combined['LotFrontage'].transform(lambda x: x.fillna(x.mode()[0]))


# In[46]:

combined['MiscFeature'] = combined['MiscFeature'].fillna('None')
combined['Exterior1st'].fillna(value= 'None', inplace = True)
combined['Exterior2nd'].fillna(value= 'None', inplace = True)
combined['Functional'].fillna(value= 'None', inplace = True)
combined['KitchenQual'].fillna(value = 'None' , inplace = True)
combined['MSZoning'].fillna(value = 'None' , inplace = True)
combined['SaleType'].fillna(value = 'None' , inplace = True)
combined['Utilities'].fillna(value = 'None' , inplace = True)
combined["MasVnrType"] = combined["MasVnrType"].fillna('None')
combined["MasVnrArea"] = combined["MasVnrArea"].fillna(0)
combined["Fence"] = combined["Fence"].fillna('None')
combined['SaleCondition'] = combined['SaleCondition'].map(lambda x: 1 if x=='Abnorml' else 0)


# In[47]:

combined[combined.columns[combined.isnull().any()]].isnull().sum()


# In[50]:

combined['SaleCondition'].unique()


# In[ ]:




# In[51]:

combined.head()


# In[52]:

#Converting Categorical variables to numeric
categorical = combined.select_dtypes(exclude=['float64', 'int64'])


# In[53]:

labelEnc=LabelEncoder()

cat_vars=['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',
       'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
       'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',
       'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',
       'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
       'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',
       'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',
       'SaleType', 'Street', 'Utilities']


# In[54]:

for col in cat_vars:
    combined[col]=labelEnc.fit_transform(combined[col])


# In[55]:

combined.head()


# In[56]:

# Year Columns
#'GarageYrBlt','YearBuilt','YearRemodAdd', 'YrSold'
combined['GarageYrBlt'].replace('None' , 100, inplace = True)


# In[57]:

combined.columns


# In[58]:

labelEnc=LabelEncoder()

cat_vars=['GarageYrBlt','YearBuilt','YearRemod/Add', 'YrSold']

for col in cat_vars:
    combined[col]=labelEnc.fit_transform(combined[col])


# In[59]:

combined.head()


# #Modelling

# In[60]:

New_Train = combined[:2051]
X_train = New_Train.drop('SalePrice',axis=1)
y_train = New_Train['SalePrice']



# In[61]:

X_train.shape


# In[62]:

New_Test = combined[2051:]
X_test = New_Test.drop('SalePrice',axis=1)


# In[63]:

X_test.shape


# In[ ]:




# In[64]:

X_train


# #Ridge Regression

# In[65]:

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# In[ ]:

# #Defining a function to calculate the RMSE for each Cross validated fold
# def rmse_cv(model):
#     rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
#     return (rmse)



# In[ ]:

# model_ridge = Ridge(alpha = 5).fit(X_train, y_train)


# In[ ]:

# alphas = [0.0001,0.1,0.5,1,2,5,7,10]
# rmse_cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
# print(rmse_cv_ridge)


# In[ ]:

# rmse_cv_ridge = pd.Series(rmse_cv_ridge, index = alphas)
# rmse_cv_ridge.plot(title = "RMSE VS Alpha")


# In[ ]:

# from sklearn.linear_model import Lasso


# In[ ]:

# model_lasso = Lasso().fit(X_train, y_train)


# In[ ]:

# alphas = [0.00001,.0001,0.001,0.002,0.005,0.01]
# rmse_cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
# print(rmse_cv_lasso)


# In[ ]:

# rmse_cv_lasso = pd.Series(rmse_cv_lasso, index = alphas)
# plt.figure(figsize=(10,4))
# rmse_cv_lasso.plot(title = "RMSE VS Alpha")


# In[ ]:

# model_lasso = Lasso(alpha = 0.001 , max_iter=1000).fit(X_train, y_train)


# In[ ]:

# rmse_cv(model_lasso).mean()


# In[ ]:

# predictors = X_train.columns

# coef = pd.Series(model_lasso.coef_, index = X_train.columns)

# imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

# plt.figure(figsize=(12,8))
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")


# In[ ]:

# lasso_preds = np.expm1(model_lasso.predict(X_test)) # reversing Log Transformation


# In[66]:

from sklearn.tree import DecisionTreeRegressorprint(reg_scores, np.mean(reg_scores))


# In[67]:

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score


# In[68]:

# cross val Linear Reg with 4 folds
reg_scores = cross_val_score(LinearRegression(), X_train, y_train, cv=4)


# In[71]:

print(reg_scores, np.mean(reg_scores))
linreg = LinearRegression().fit(X_train, y_train)


# In[ ]:

# set 4 models
dtr1 = DecisionTreeRegressor(max_depth=1)
dtr2 = DecisionTreeRegressor(max_depth=2)
dtr3 = DecisionTreeRegressor(max_depth=3)
dtrN = DecisionTreeRegressor(max_depth=None)


# In[72]:

# fit the 4 models
dtr1.fit(X_train, y_train)
dtr2.fit(X_train, y_train)
dtr3.fit(X_train, y_train)
dtrN.fit(X_train, y_train)


# In[73]:

# cross validate the 4 models
dtr1_scores = cross_val_score(dtr1, X_train, y_train, cv=4)
dtr2_scores = cross_val_score(dtr2, X_train, y_train, cv=4)
dtr3_scores = cross_val_score(dtr3, X_train, y_train, cv=4)
dtrN_scores = cross_val_score(dtrN, X_train, y_train, cv=4)

# score the 4 models
print(dtr1_scores, np.mean(dtr1_scores))
print(dtr2_scores, np.mean(dtr2_scores))
print(dtr3_scores, np.mean(dtr3_scores))
print(dtrN_scores, np.mean(dtrN_scores))


# In[80]:

dt_preds = DecisionTreeRegressor.predict(dtrN,X_test,check_input=True)


# In[ ]:




# In[ ]:




# In[85]:

submission1 = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": dt_preds
    })


# In[87]:

submission1.to_csv("HousePrice1.csv", index=False)


# #####Classification Model######

# In[ ]:

X_train['SaleCondition'].value_counts()


# In[ ]:



