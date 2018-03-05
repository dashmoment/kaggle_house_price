
'''
Competion path:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    
Kernel url:
    https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

Dataset path:
    /home/ubuntu/dataset/kaggle/house_price

Workflow:
        

    * Imputing missing values by proceeding sequentially through the data

    * Transforming some numerical variables that seem really categorical

    * Label Encoding some categorical variables that may contain information in their ordering set

    * Box Cox Transformation of skewed features (instead of log-transformation) : This gave me a slightly better result both on leaderboard and cross-validation.

    * Getting dummy variables for categorical features.

'''

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


#Load train and test set
data_path = '/home/ubuntu/dataset/kaggle/house_price'
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
print('Train set head 5:',train.head(5))
print("Train set:", train.shape)
print("Test set:", test.shape)

#Data prerpocessing

#Find outlier
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Foud two data have large living size but low price --> Drop outlier data
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

'''
Outlier should be check for every numerical data
But temp ignore here
'''

'''
Sale price analysis
'''
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()



#Features engineering  --> Combine train and test for feature extracting
#1. Features engineering - Data mining

#1-1Transform sale price by loglp to checkout if the sale price fit normal distribution
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice_log"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice_log'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice_log'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice_log'], plot=plt)
plt.show()

y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice', 'SalePrice_log'], axis=1, inplace=True)
print(all_data.shape)


#1-2 find NA data and do some statistic
is_all_data_na = all_data.isnull().mean() #percentage of missing data
all_data_na = is_all_data_na.drop(is_all_data_na[is_all_data_na==0].index)
missing_ratio = all_data_na.sort_values(ascending=False)
missing_ratio.head(30)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_ratio.index, y=missing_ratio)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#1-3 feature corelation

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
corrmat_SalePrice_log = abs(corrmat['SalePrice_log'])
corrmat_SalePrice_log_sort = corrmat_SalePrice_log.sort_values(ascending=False)
print(corrmat_SalePrice_log_sort[corrmat_SalePrice_log_sort > 0.5])


#Imputing missing value
#fill None if na feature means house has no feature. e.g. PoolQC is NA
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


### LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , 
### we can fill in missing values by the median LotFrontage of the neighborhood.
### Better idea????

print(corrmat['LotFrontage'].sort_values(ascending=False))
LotFrontage_gby_Neighborhood = all_data.groupby("Neighborhood")["LotFrontage"].apply(list)

L_gby_N_drop = dict()
for idx in LotFrontage_gby_Neighborhood.index:
    L_gby_N_drop[idx] = np.median([x for x in LotFrontage_gby_Neighborhood[idx] if pd.isnull(x) == False])
    
L_gby_N_drop = pd.Series(L_gby_N_drop)
plt.xticks(rotation='90')
sns.barplot(x=L_gby_N_drop.index, y=L_gby_N_drop.values)
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#Basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

#masonry veneer 
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# MSZoning (The general zoning classification)
MSZoning_drop_na = all_data["MSZoning"].drop(all_data["MSZoning"].isnull == True)
sns.countplot(x=MSZoning_drop_na.index, data=MSZoning_drop_na)

#Since RL is the most common class, fill N/A by RL, because missing ratio is samll (0.001371)
#Pandas mode : list most common class
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data['MSZoning'].mode()[0])
print(missing_ratio['MSZoning'])

#Utilities
Utilities_drop_na = all_data["Utilities"].drop(all_data["Utilities"].isnull == True)
sns.countplot(x=Utilities_drop_na.index, data=Utilities_drop_na)
print(missing_ratio['Utilities'])

#Since missing ratio is small (0.00068) and most of data is the same class, so drop this feature
all_data = all_data.drop(['Utilities'], axis=1)

