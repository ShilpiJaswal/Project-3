
# coding: utf-8

# In[196]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# In[197]:

#Reading in our dataset
train_df = pd.read_csv('train.csv') # reading in the training dataset.
test_df = pd.read_csv('test.csv')# reading in the test dataset.
combined = pd.concat([train_df,test_df], axis=0) #combining train and test dataset for data preprocessing
combined.shape


# In[198]:

combined.head()


# In[199]:

combined['Sale Condition'].describe()


# In[200]:

combined['Sale Condition']


# In[201]:

combined.columns = [x.replace(' ','') for x in combined.columns]


# In[202]:

combined['SaleCondition'].value_counts()


# In[203]:

combined['SaleCondition'] = combined['SaleCondition'].map(lambda x: 1 if x=='Abnorml' else 0)


# In[204]:

combined['SaleCondition'].value_counts()


# In[205]:

mean_corr = combined.drop('Id', axis=1).corr()

# Set the default matplotlib figure size:
fig, ax = plt.subplots(figsize=(18,10))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(mean_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(mean_corr, mask=mask, ax=ax)

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)

# If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
plt.show()


# In[206]:

#Missing Value imputation
#Categorical variables
combined.select_dtypes(include=['object']).columns


# In[207]:

#Numerical Columns
combined.select_dtypes(include=['float64', 'int64']).columns


# In[208]:

combined['OverallQual'].isnull().sum()


# In[ ]:




# In[ ]:










# In[209]:

combined['CentralAir'] = combined['CentralAir'].map(lambda x: 1 if x=='Y' else 0)


# In[210]:

combined['Street'] = combined['Street'].map(lambda x: 1 if x=='Pave' else 0)


# In[211]:

combined['Street'].value_counts()


# In[214]:

New_Train = combined[:2051]
X_train = New_Train.drop('SaleCondition',axis=1)
y_train = New_Train[['Id','SaleCondition']]


# In[252]:

New_Test = combined[2051:]
X_test = New_Test.drop('SaleCondition',axis=1)
y_test = New_Test['SaleCondition']


# In[253]:

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# In[254]:

y_test


# In[264]:

y = y_train
X = X_train[['Id','YearBuilt','YrSold','OverallCond','OverallQual','CentralAir','LotArea','Street']]


# In[265]:

X_test = X_test[['Id','YearBuilt','YrSold','OverallCond','OverallQual','CentralAir','LotArea','Street']]


# In[266]:


# Function to crossvalidate accuracy of a knn model acros folds
def accuracy_crossvalidator(X, y, knn, cv_indices):
    
    # list to store the scores/accuracy of folds
    scores = []
    
    # iterate through the training and testing folds in cv_indices
    for train_i, test_i in cv_indices:
        
        # get the current X train & test subsets of X
        X_train = X[train_i, :]
        X_test = X[test_i, :]

        # get the Y train & test subsets of Y
        Y_train = y[train_i]
        Y_test = y[test_i]

        # fit the knn model on the training data
        knn.fit(X_train, Y_train)
        
        # get the accuracy predicting the testing data
        acc = knn.score(X_test, Y_test)
        scores.append(acc)
        
        print(('Fold accuracy:', acc))
        
    print(('Mean CV accuracy:', np.mean(scores)))
    return scores


# In[267]:

baseline =1 - np.mean(y['SaleCondition'])
print('baseline:', baseline)


# In[271]:

knn = KNeighborsClassifier()
knn.fit(X_test,y_test) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
predict = knn.predict(X_test)


# In[272]:

predict


# In[ ]:




# In[ ]:




# ####Logistic regression

# In[270]:

from sklearn.linear_model import LogisticRegressionCV

Cs = np.logspace(-4, 4, 100)
lr = LogisticRegressionCV(Cs=Cs, cv=5)
lr.fit(X, y)

print(lr.C_)


# In[105]:

from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(lr, Xs, y, cv=5))


# In[106]:

from sklearn.tree import DecisionTreeClassifier


# In[110]:

dtc1 = DecisionTreeClassifier(max_depth=1)
# dtc2 = DecisionTreeClassifier(max_depth=2)
# dtc3 = DecisionTreeClassifier(max_depth=3)
# dtcN = DecisionTreeClassifier(max_depth=None)


# In[108]:

dtc1.fit(X, y)
# dtc2.fit(X, y)
# dtc3.fit(X, y)
# dtcN.fit(X, y)


# In[111]:

# use CV to evaluate the 4 trees
dtc1_scores = cross_val_score(dtc1, X, y, cv=4)
# dtc2_scores = cross_val_score(dtc2, X, y, cv=4)
# dtc3_scores = cross_val_score(dtc3, X, y, cv=4)
# dtcN_scores = cross_val_score(dtcN, X, y, cv=4)

print(dtc1_scores, np.mean(dtc1_scores))
# print(dtc2_scores, np.mean(dtc2_scores))
# print(dtc3_scores, np.mean(dtc3_scores))
# print(dtcN_scores, np.mean(dtcN_scores))


# In[276]:

score = pd.DataFrame({
        "Id": X_test["Id"],
        "Sale Condition":predict
    })


# In[278]:

score.to_csv("HousePriceClassifier.csv", index=False)


# In[ ]:



