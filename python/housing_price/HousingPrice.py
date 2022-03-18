#!/usr/bin/env python
# coding: utf-8

# # Housing price project
# 
# The prediction of housing price is a regression problem.
# 
# In this notebook, I am working on prediction of housing price in which I use a data set fetched from Kaggle. This data set includes both numeric and non-numeric features.
# 
# At first, I'll only take numerical features into account and try to trains model only based off them.
# Later I'll try to solve the problem while considering both numerical and non-numerical features of data. 
# 
# 
# 

# Prediction contains different steps:
# 1) Data prepreocessing,
# 
# 2) Data cleaning,
# 
#     1) Impute
#     2) Scaling
#     3) 
#     
# 3) ML model

# In[46]:


# import common libraries
import numpy as np
import pandas as pd
import copy


# In[47]:

if __name__ == '__main__':

    # Read the data set from files inside the same folder as this notebook
    trainset = pd.read_csv('trainHP.csv')
    testset = pd.read_csv('testHP.csv')

    # to avoid ruining the original data
    dftrain = copy.deepcopy(trainset)
    dftest = copy.deepcopy(testset)

    # See the dimension of the dataset
    print("trainset shape:", dftrain.shape)
    print("testset shape:", dftest.shape)


    # ### Extract features and target from data set

    # In[48]:


    features = dftrain.drop('SalePrice',axis=1,inplace=False)
    target = dftrain['SalePrice']


    # In[49]:


    # Due to the metric by Kaggle we take the logarithm of the price
    Y = np.log(target)


    # ### Seperate numeric and non-numeric columns in features

    # In[50]:


    # Select numeric columns from data frame
    features_numeric = features.select_dtypes(include=[np.number])
    # the line below gives a numpy array including column names of numeric features.
    numcol = features_numeric.columns
    #print(type(features_numeric.columns.values))

    # select non-numeric columns from data frame
    features_nonnumeric = features.select_dtypes(exclude=[np.number])
    noncol = features_nonnumeric.columns.values
    #print('a list of nonnumeric features:')


    # In[51]:


    noncol


    # ### Observe numeric data set

    # In[52]:


    features_numeric.info()


    # In[53]:


    features_numeric.head(10)


    # As you can see in the data frame above, some of the samples contain a `NaN` as value for a feature or more. `NaN` is a place holder for missing values, values that are absent in the datasent for some reason.

    # In[54]:


    Null=features_numeric.isnull().sum()
    print("The number of columns including missing values in numeric features is ", Null[Null>0].count())
    Null[Null>0]


    # ### Data Visualization

    # In[55]:


    import matplotlib.pyplot as plt


    # In[56]:


    # plt.hist(features_nonnumeric['Street'])


    # In[57]:


    # features_numeric.hist(bins=30,figsize=(20,20))
    # plt.show()


    # In[58]:


    # plt.hist(features_nonnumeric.MSZoning)


    # ### Data cleaning

    # The simplest solution is to drop missing values. However, I decided to fill the missing values with mean value of corresponding column.

    # In[59]:


    from sklearn.impute import SimpleImputer

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    # the following line gives a numpy arry
    features_numeric_imputed = imp_mean.fit_transform(features_numeric)

    # we should make it as pandas Data Frame
    features_numeric_imputed = pd.DataFrame(features_numeric_imputed, columns=numcol)

    features_numeric_imputed.head(10)


    # Finally, again I check numeric columns looking for further missing value, expecting none.

    # In[60]:


    Null=features_numeric_imputed.isnull().sum()
    Null[Null>0].count()


    # ### Feature Scaling
    # There are two methods:
    # * StandardScaler: makes the mean of observed values zero and standard deviation one. Resulted Data will have both negative and positive values.
    #
    #     * It's more benefitial for datasets with inputs having different scales.
    #
    # * MinMax: maps each value into a new value ranging from 0 to 1
    #
    #
    # ! Please note that the following code snippets marked with a Star* are alternatives. ONLY ONE of them should be executed.

    # ### * Data Standarization (StandardScaler)

    # In[61]:


    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # The line below gives numpy array,
    features_numeric_scaled = scaler.fit_transform(features_numeric_imputed)

    # Create a pandas Data Frame
    features_numeric_scaled = pd.DataFrame(features_numeric_scaled, columns=numcol)

    features_numeric_scaled.head(10)


    # ### * Data Normalization (MinMax)

    # In[62]:


    from sklearn.preprocessing import MinMaxScaler


    # In[63]:


    scaler = MinMaxScaler()
    numdataren = scaler.fit_transform(features_numeric_imputed)
    numdataren.size
    dfscaled = pd.DataFrame(numdataren, columns = numcol)
    dfscaled


    # In[64]:


    # split train set into random train and test subsets.
    from sklearn.model_selection import train_test_split

    X_train, X_valid, Y_train, Y_valid = train_test_split(features_numeric_scaled, Y, train_size=0.7, test_size=0.3)

    print(f"No. of training examples: {X_train.shape[0]}")
    print(f"No. of testing examples: {X_valid.shape[0]}")


    # In[65]:


    # prediction model
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(random_state=1)

    # implement model on X_train, Y_train
    model.fit(X_train, Y_train)

    # prediction
    Y_pre = model.predict(X_valid)
    print("BMD::X_Valid\n", X_valid.iloc[0])
    Y_pre


    # In[66]:


    from sklearn.metrics import mean_absolute_error

    mean_absolute_error(Y_valid, Y_pre)


    # ### Marshaling the model into a file

    # In[67]:


    from sklearn2pmml import PMMLPipeline, sklearn2pmml

    pipeline = PMMLPipeline([('regressor', DecisionTreeRegressor(random_state=1))])
    pipeline.fit(X_train, Y_train)
    Y_pre = pipeline.predict(X_valid)
    Y_pre
    sklearn2pmml(pipeline, 'model.pmml', with_repr=True, debug=True)

