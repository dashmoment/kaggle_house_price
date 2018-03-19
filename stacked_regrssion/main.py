
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
#data_path = '/home/ubuntu/dataset/kaggle/house_price'
data_path = '../data'
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
print('Train set head 5:',train.head(5))
print("Train set:", train.shape)
print("Test set:", test.shape)

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

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


######################
#Features engineering  --> Combine train and test for feature extracting
#1. Features engineering - Data mining
######################

#1-1Transform sale price by loglp to checkout if the sale price fit normal distribution
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
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

y_train = train.SalePrice_log.values
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

######################
#2. Imputing missing value
######################
#fill None if na feature means house has no feature. e.g. PoolQC is NA
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#MiscFeature : data description says NA means "no misc feature"
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

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

#Functional : data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual: Only one NA value, and same as Electrical, we set 'TA' 
#(which is the most frequent) for the missing value in KitchenQual.
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#SaleType : Fill in again with most frequent which is "WD"
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

#MSSubClass : Na most likely means No building class. We can replace missing values with None
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check if there still are missing feature
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

######################
#3. Transforming some numerical variables that are really categorical
######################
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

######################
#4. Label Encoding some categorical variables that may contain information in their ordering set
######################


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


######################
#5. Adding total sqfootage feature 
#Since area related features are very important to determine house prices, 
#we add one more feature which is the total area of basement, 
#first and second floor areas of each house
######################

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

####################
#6. Check Skew feature
###################
# Check the skew of all numerical features, Categortical type is object
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

#Display skew feature as example
sns.distplot(all_data['MiscVal'])
sns.distplot(all_data['PoolQC']) 

#Box Cox Transformation of (highly) skewed features
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

#Check results of BoxCox
sns.distplot(all_data['MiscVal'])
sns.distplot(all_data['PoolQC'])    
    
####################
#7. Getting dummy categorical features
####################
all_data = pd.get_dummies(all_data)
print(all_data.shape)

####################
#8. Finish feature engineering, Split train/test set
####################
train = all_data[:ntrain]
test = all_data[ntrain:]

####################
#9. Modeling 
####################
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


#LASSO Regression 
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
#Gradient Boosting Regression
Boost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#Save to submission files

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def submission(pred, model, test_ID=test_ID):
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = pred
    save_filename = 'submission_' + model + '.csv' 
    sub.to_csv(save_filename,index=False)
    print(pred)
    

def fit_model_and_submission(model, model_name, x_train = train, y_train = y_train, x_test=test, test_ID=test_ID):
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    rmse = rmsle(y_train, pred_train)
    print("rmse_",model_name," : ", rmse)
    submission(np.expm1(pred_test), model_name)
   
    


score = rmsle_cv(ENet)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

fit_model_and_submission(lasso, 'lasso')
fit_model_and_submission(ENet, 'ENet')
fit_model_and_submission(KRR, 'KRR')
fit_model_and_submission(Boost, 'Boost')
fit_model_and_submission(model_xgb, 'xgb')
fit_model_and_submission(model_lgb, 'lgb')

#Simple average model:
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
averaged_models = AveragingModels(models = (ENet, Boost, KRR, lasso))
fit_model_and_submission(averaged_models, 'simpleAVG')



#Stacking average model class

base_model = [ENet, Boost, KRR,model_xgb]
#base_model = [model_xgb]
meta_model = clone(lasso)
n_fold = 5

base_models_ = [list() for x in base_model]
kfold = KFold(n_splits= n_folds, shuffle=True, random_state=156)
out_of_fold_predictions = np.zeros((train.shape[0], len(base_model)))

for i, model in enumerate(base_model):
    for train_index, holdout_index in kfold.split(train.values, y_train):
        instance = clone(model)
        base_models_[i].append(instance)
        instance.fit(train.values[train_index], y_train[train_index])
        y_pred = instance.predict(train.values[holdout_index])
        out_of_fold_predictions[holdout_index, i] = y_pred
                
      
meta_model.fit(out_of_fold_predictions, y_train)


def stacked_model_prediction(X):
#Prediction of Stacking average model
#Avergae prediction results of models of 5 folds
#For each models, it derive 5 sub models which are trained by different folds
    meta_feature = np.column_stack([np.column_stack([model.predict(X.values) for model in base_models]).mean(axis=1)
                for base_models in base_models_ ])
    
    return meta_model.predict(meta_feature)



#pack stacked model into class for fit with other module
    
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        
        X = X.values
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X.values) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = ([ENet, Boost, KRR,model_xgb]),
                                                 meta_model = lasso)

fit_model_and_submission(stacked_averaged_models, 'stackAVG')

#Esemble XGBoost, LightGBM, stacked_averaged_models

stacked_averaged_models_predict = stacked_averaged_models.predict(train)
model_xgb_predict = model_xgb.predict(train)
model_lgb_predict = model_lgb.predict(train)
ens_predict =  stacked_averaged_models_predict*0.70 + model_xgb_predict*0.15 + model_lgb_predict*0.15 
print('Stacked RMSE: {}'.format(rmsle(y_train,stacked_averaged_models_predict)))
print('XGBoost RMSE: {}'.format(rmsle(y_train,model_xgb_predict)))
print('Ensemble RMSE: {}'.format(rmsle(y_train,ens_predict)))

stacked_averaged_models_predict = stacked_averaged_models.predict(test)
model_xgb_predict = model_xgb.predict(test)
ens_predict =  stacked_averaged_models_predict*0.70 + model_xgb_predict*0.15 + model_xgb_predict*0.15 
submission(ens_predict, 'Ensemble')




