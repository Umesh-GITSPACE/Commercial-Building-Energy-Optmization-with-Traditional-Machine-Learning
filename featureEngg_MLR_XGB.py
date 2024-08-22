# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:56:41 2024

@author: umesh
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor
#Read files
energy_use_data=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/ele.csv')
out_env_data=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/site_weather.csv')
in_cooling_data=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/zone_temp_sp_c.csv')
in_heating_data=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/zone_temp_sp_h.csv')
zone_temp_interior=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/zone_temp_interior.csv')
zone_temp_exterior=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/zone_temp_exterior.csv')
co2_conc=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/zone_co2.csv')
occupant=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/occ.csv')
wifi_data=pd.read_csv('D:/MSDA_SJSU/Data 245 ML/Project/Bldg59_clean data/wifi.csv')
energy_use_target=energy_use_data[['date','hvac_N','hvac_S']]
energy_use_target.date


energy_use_target['date'] = pd.to_datetime(energy_use_target['date'],format='%Y/%m/%d %H:%M')
energy_use_target['date'] = pd.to_datetime(energy_use_target['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
in_cooling_data['date'] = pd.to_datetime(in_cooling_data['date'],format='%Y/%m/%d %H:%M')
in_cooling_data['date'] = pd.to_datetime(in_cooling_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
in_heating_data['date'] = pd.to_datetime(in_heating_data['date'],format='%Y/%m/%d %H:%M')
in_heating_data['date'] = pd.to_datetime(in_heating_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
co2_conc['date'] = pd.to_datetime(co2_conc['date'],format='%Y/%m/%d %H:%M')
co2_conc['date'] = pd.to_datetime(co2_conc['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
wifi_data['date'] = pd.to_datetime(wifi_data['date'],format='%Y/%m/%d %H:%M')
wifi_data['date'] = pd.to_datetime(wifi_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
##Converting all date columns into date formatted type
out_env_data['date'] = pd.to_datetime(out_env_data['date'],format='%Y-%m-%d %H:%M:%S')
zone_temp_interior['date'] = pd.to_datetime(zone_temp_interior['date'],format='%Y-%m-%d %H:%M:%S')
zone_temp_exterior['date'] = pd.to_datetime(zone_temp_exterior['date'],format='%Y-%m-%d %H:%M:%S')
occupant['date'] = pd.to_datetime(occupant['date'],format='%Y-%m-%d %H:%M:%S')
def change_dateformat(dataframe,col):
    dataframe[col]=pd.to_datetime(dataframe[col],format=='%Y/%m/%d %H:%M')
    dataframe[col]=dataframe[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return dataframe
in_cooling_data.drop(columns=['Unnamed: 42','Unnamed: 43','Unnamed: 44','Unnamed: 45',
                            'Unnamed: 46',
                            'Unnamed: 47', 
                            'Unnamed: 48',
                            'Unnamed: 49',
                            'Unnamed: 50',
                            'Unnamed: 51'],inplace=True)
in_heating_data.drop(columns=['Unnamed: 42','Unnamed: 43','Unnamed: 44','Unnamed: 45',
                            'Unnamed: 46',
                            'Unnamed: 47', 
                            'Unnamed: 48',
                            'Unnamed: 49',
                            'Unnamed: 50',
                            'Unnamed: 51'],inplace=True)
in_heating_data.head()
#Merged_df=pd.merge(out_env_data,in_cooling_data,in_heating_data,zone_temp_interior,zone_temp_exterior,co2_conc,occupant,how='left')
#Merging indoor cooling and heating data
Merge_cooling_heating=pd.merge(in_cooling_data,in_heating_data,how='left')
#Merging outdoor environment data with indoor merged cooling and heating data
Merge_indoor_outdoor=pd.merge(out_env_data,Merge_cooling_heating,how='left')
#Merging with energy_use_target data
Merge_en=pd.merge(energy_use_target,Merge_indoor_outdoor,how='left')
Merge_en.isnull().sum()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn import linear_model

#imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
Merge_en.columns
X=Merge_en.loc[:,'hvac_N':'zone_071_heating_sp']
X.head()
# define imputer
imputer = IterativeImputer(random_state=42)
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
Xtrans=pd.DataFrame(Xtrans.tolist())
#Replacing transformed columns backinto Merge_indoor_outdoor dataframe 
Xtrans.columns=X.columns
Merge_en.loc[:,'hvac_N':'zone_071_heating_sp']=Xtrans
#Percentage of null values in each columns
Merge_en.isnull().sum() * 100 / len(Merge_en)
#Merging interior and exterior zone temperature data
Merge_zone_temp=pd.merge(zone_temp_interior, zone_temp_exterior,how='left')
#Merging zone temperature with CO2 concentration
Merge_zonetemp_co2=pd.merge(Merge_zone_temp,co2_conc,how='left')
#Merging zone temperature with CO2 concentration
#Merge_zonetemp_co2=pd.merge(Merge_zone_temp,Merge_zone_temp,how='left')
Merge_zonetemp_co2.head()
#Percentage of null values in each columns
Merge_zonetemp_co2.isnull().sum() * 100 / len(Merge_zonetemp_co2)
feature_Trans1=Merge_zonetemp_co2.loc[:,'cerc_templogger_1':'zone_072_co2']
feature_Trans1.columns
iter_imputer = IterativeImputer(random_state=42)
# fit on the dataset
iter_imputer.fit(feature_Trans1)
Xtrans1=iter_imputer.transform(feature_Trans1)
Xtrans1=pd.DataFrame(Xtrans1.tolist())
Xtrans1
Xtrans1.columns=feature_Trans1.columns
Xtrans1.head()

Merge_zonetemp_co2.loc[:,'cerc_templogger_1':'zone_072_co2']=Xtrans1
Merge_zonetemp_co2.isnull().sum()
all_df=pd.merge(Merge_en, Merge_zonetemp_co2,how='inner')
all_df.head()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#to perform PCA we are taking numerical fields only
all_df_num=all_df.loc[:,'air_temp_set_1':]
# Step 1: Standardize the Data
scaler = StandardScaler()
all_df_num_std = scaler.fit_transform(all_df_num)
# Step 2-5: PCA
pca = PCA()
all_df_pca = pca.fit_transform(all_df_num_std)
#Important components from PCA
pd.DataFrame(abs(pca.components_)).shape
# Plot Explained Variance Ratio
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)
plt.figure(figsize=(5,4),dpi=150)
plt.grid()
plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Principal Components')
plt.show()
# example of correlation feature selection for numerical data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
all_df.drop(columns=['date'],inplace=True)
feature=all_df.drop(columns=['hvac_S','hvac_N'])
target=all_df[['hvac_S','hvac_N']]
feature.head()
# spliting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.33, random_state=1)
# feature selection
def select_features_corstat(X_train, y_train, X_test):
    #Converting y_train into 1-dimensional
    y_train=np.ravel(y_train)
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# feature selection process
X_train_fs, X_test_fs, fs = select_features_corstat(X_train, y_train.hvac_S, X_test)
# Checking the scores for the features
feature_list_hvacS_cor={}
for i in range(len(fs.scores_)):
    feature_list_hvacS_cor.update({fs.feature_names_in_[i]: fs.scores_[i]})
    print('Feature %d: %s : %f' % (i,fs.feature_names_in_[i], fs.scores_[i]))
#creating dataframe from dictionary
selected_feature_1=pd.DataFrame.from_dict(feature_list_hvacS_cor,columns=['value'],orient='index')

#Sorting and listing top 40 features based on scores
top_40_withcorr_score_S=selected_feature_1.sort_values(by='value',ascending=False)[:40]
top_40_withcorr_score_S.index
feature_select_df_hvacS_1=all_df.loc[:,top_40_withcorr_score_S.index]
#feature_select_df_hvacS_1.to_csv('feature_select_df_hvacS_1.csv', sep=',', index=False, encoding='utf-8')
# plot the scores
#plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.figure(figsize=(25, 8))
plt.bar(selected_feature_1.index,selected_feature_1.value)
plt.xticks(selected_feature_1.index, rotation='vertical')
plt.show()
# Building Linear Regression model
Ln_model = LinearRegression()
# Training the model
Ln_model.fit(X_train_fs, y_train.hvac_S)
# Evaluating the model
ypred_lin = Ln_model.predict(X_test_fs)
# evaluating predictions with MAE
mae = mean_absolute_error(y_test.hvac_S, ypred_lin)
print('MAE: %.3f' % mae)
# feature selection process
X_train_fsN, X_test_fsN, fsN = select_features_corstat(X_train, y_train.hvac_N, X_test)
# Checking the scores for the features
feature_list_hvacN_cor={}
for i in range(len(fsN.scores_)):
    feature_list_hvacN_cor.update({fsN.feature_names_in_[i]: fsN.scores_[i]})
    print('Feature %d: %s : %f' % (i,fsN.feature_names_in_[i], fsN.scores_[i]))
#creating dataframe from dictionary
selected_feature_N1=pd.DataFrame.from_dict(feature_list_hvacN_cor,columns=['value'],orient='index')

#Sorting and listing top 40 features based on scores
top_40_withcorr_score_N=selected_feature_N1.sort_values(by='value',ascending=False)[:40]
# plot the scores

plt.figure(figsize=(25, 8))
plt.bar(selected_feature_N1.index,selected_feature_N1.value)
plt.xticks(selected_feature_N1.index, rotation='vertical')
plt.title('Correlation Statistics for the all Features -target variable Hvac_N')
plt.show()
# feature selection
def select_features_mutinfo(X_train, y_train, X_test):
    #Converting y_train into 1-dimensional
    y_train=np.ravel(y_train)
    # configure to select a subset of features
    fs = SelectKBest(score_func=mutual_info_regression, k=88)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# feature selection
X_train_fs, X_test_fs, fs = select_features_mutinfo(X_train, y_train.hvac_S, X_test)
# Checking the scores for the features
feature_list_hvacS_mut={}
for i in range(len(fs.scores_)):
    feature_list_hvacS_mut.update({fs.feature_names_in_[i]: fs.scores_[i]})
    print('Feature %d: %s : %f' % (i,fs.feature_names_in_[i], fs.scores_[i]))
    #creating dataframe from dictionary
selected_feature_2=pd.DataFrame.from_dict(feature_list_hvacS_mut,columns=['value'],orient='index')

#Sorting and listing top 40 features based on scores
top_40_withmut_score=selected_feature_2.sort_values(by='value',ascending=False)[:40]

feature_select_df_hvacS_2=all_df.loc[:,top_40_withmut_score.index]
#feature_select_df_hvacS_2.to_csv('feature_select_df_hvacS_2.csv', sep=',', index=False, encoding='utf-8')
top_40_withmut_score.index
# plot the scores

plt.figure(figsize=(25, 8))
plt.bar(selected_feature_2.index,selected_feature_2.value)
plt.xticks(selected_feature_2.index, rotation='vertical')
plt.title('Mutual Information Score for the all Features -target variable Hvac_S')
plt.show()
# fiting the model
Ln_model_2 = LinearRegression()
Ln_model_2.fit(X_train_fs, y_train)
# evaluate the model
y_pred_2 = Ln_model_2.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_pred_2, y_pred_2)
print('MAE: %.6f' % mae)
# feature selection
X_train_fsN, X_test_fsN, fsN = select_features_mutinfo(X_train, y_train.hvac_N, X_test)
# Checking the scores for the features
feature_list_hvacN_mut={}
for i in range(len(fsN.scores_)):
    feature_list_hvacN_mut.update({fsN.feature_names_in_[i]: fsN.scores_[i]})
    print('Feature %d: %s : %f' % (i,fsN.feature_names_in_[i], fsN.scores_[i]))
#creating dataframe from dictionary
selected_feature_N2=pd.DataFrame.from_dict(feature_list_hvacN_mut,columns=['value'],orient='index')

#Sorting and listing top 40 features based on scores
top_40_withmut_score_N=selected_feature_N2.sort_values(by='value',ascending=False)[:40]
top_40_withmut_score_N.index
# plot the scores

plt.figure(figsize=(25, 8))
plt.bar(selected_feature_N2.index,selected_feature_N2.value)
plt.xticks(selected_feature_N2.index, rotation='vertical')
plt.title('Mutual Information score for the all Features -target variable Hvac_N')
plt.show()
#Selected features through Cooreleation Statistics for target Hvac_S : top_40_withcorr_score_S.index
feature_corr=feature.loc[:,top_40_withcorr_score_S.index]
#data=feature_corr
#data.to_csv('data_X_all_corr.csv', index=False) 
#target.to_csv('target_all_corr.csv',index=False)
##We will keep 10% data for demo purpose.
X1, X_test_demo, y1, y_test_demo = train_test_split(feature_corr,target, random_state=42, test_size=0.1)
## Saving the data for Demo
X_test_demo.to_csv('X_test_corr_demo.csv', index=False)  
y_test_demo.to_csv('y_test_corr_demo.csv', index=False)  
## Saving the data for demo modelling
X1.to_csv('X_train_corr_demo.csv', index=False) 
y1.to_csv('y_train_corr_demo.csv', index=False) 
#Splitting the data in 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X1,y1, random_state=42, test_size=0.3)
## Function to evaluate the regression models
def evaluate_regression_model(y_test,y_pred):
    print('R2 score: {}'.format(r2_score(y_test,y_pred)))
    print('Mean Absolute Error (MAE): {}'.format(mean_absolute_error(y_test, y_pred)))
    print('Mean Squared Error (MSE): {}'.format(mean_squared_error(y_test, y_pred)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): {}'.format(rmse))
    print('Mean Absolute Percentage Error (MAPE): {}'.format(mean_absolute_percentage_error(y_test, y_pred)))
#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', RobustScaler()),
  #  ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))
#Predicting the target variables
pred_cor_y=pipeline.predict(X_test)
#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y)
#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=3, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))
#Predicting the target variables
pred_cor_y_ridge=ridge_pipeline.predict(X_test)
#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_ridge)
#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_ridge)
#Selected features through Cooreleation Statistics for target Hvac_N : top_40_withcorr_score_N.index
feature_corr_N=feature.loc[:,top_40_withcorr_score_N.index]
##We will keep 10% data for demo purpose.
X2, X_test_demo2, y2, y_test_demo2 = train_test_split(feature_corr_N,target, random_state=42, test_size=0.1)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X2,y2, random_state=42, test_size=0.3)

#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_cor_y=pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y)
#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=3, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_cor_y_ridge=ridge_pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=.5, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_cor_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_lasso)

 
feature_mut=feature.loc[:,top_40_withmut_score.index]
##We will keep 10% data for demo purpose.
X3, X_test_demo3, y3, y_test_demo3 = train_test_split(feature_mut,target, random_state=42, test_size=0.1)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X3,y3, random_state=42, test_size=0.3)

#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', StandardScaler()),
 #   ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y=pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y)

#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=100, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_ridge=ridge_pipeline.predict(X_test)
#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=.1, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_lasso)

#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=3, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_cor_y_ridge=ridge_pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=.5, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_cor_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_cor_y_lasso)

feature_mut=feature.loc[:,top_40_withmut_score.index]
##We will keep 10% data for demo purpose.
X3, X_test_demo3, y3, y_test_demo3 = train_test_split(feature_mut,target, random_state=42, test_size=0.1)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X3,y3, random_state=42, test_size=0.3)

#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', StandardScaler()),
 #   ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y=pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y)

#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=100, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_ridge=ridge_pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=.1, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_lasso)

#Selected features through Mutual Information Statistics for target Hvac_N : top_40_withmut_score_N.index

feature_mut_N=feature.loc[:,top_40_withmut_score_N.index]
##We will keep 10% data for demo purpose.
X4, X_test_demo4, y4, y_test_demo4 = train_test_split(feature_mut_N,target, random_state=42, test_size=0.1)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X4,y4, random_state=42, test_size=0.3)

#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', StandardScaler()),
  #  ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y=pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y)

#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', RobustScaler()),
 #   ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=0.06, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_ridge=ridge_pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=100, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_lasso)

#------------------- End of Multiple Linear Regression Model ---------

#Selected features through Cooreleation Statistics for target Hvac_S : top_40_withcorr_score_S.index
feature_corr_S=feature.loc[:,top_40_withcorr_score_S.index]

##We will keep 10% data for demo purpose.
X5, X_test_demo5, y5, y_test_demo5 = train_test_split(feature_corr_S,target, random_state=42, test_size=0.1)
                                                      
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X5,y5.hvac_S, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo5.to_csv('X_test_XGBcorrS_demo.csv', index=False)  
y_test_demo5.to_csv('y_test_XGBcorrS_demo.csv', index=False)  
## Saving the data for demo modelling
X5.to_csv('X_train_XGBcorrS_demo.csv', index=False) 
y5.to_csv('y_train_XGBcorrS_demo.csv', index=False)
# Building the model
xgb_corrS = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=5)
grid_search_corrS = GridSearchCV(xgb_corrS, param_grid, cv=kf)
grid_search_corrS.fit(X_train, y_train)
print('Best Parameters : ', grid_search_corrS.best_params_)
print('Best Score : ', grid_search_corrS.best_score_)

#Predicting the target variables
pred_corrS=grid_search_corrS.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_corrS)

Pkl_Filename_HvacS = "Pickle_XGBwithCorrS_Model.pkl"  

with open(Pkl_Filename_HvacS, 'wb') as file:  
    pickle.dump(grid_search_corrS, file)

#Selected features through Cooreleation Statistics for target Hvac_N : top_40_withcorr_score_N.index
feature_corr_N=feature.loc[:,top_40_withcorr_score_N.index]

##We will keep 10% data for demo purpose.
X6, X_test_demo6, y6, y_test_demo6 = train_test_split(feature_corr_N,target, random_state=42, test_size=0.1)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X6,y6.hvac_N, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo6.to_csv('X_test_corrN_demo.csv', index=False)  
y_test_demo6.to_csv('y_test_corrN_demo.csv', index=False)  
# Building the model
xgb_corrN = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_corrN = GridSearchCV(xgb_corrN, param_grid, cv=kf)
grid_search_corrN.fit(X_train, y_train)
print('Best Parameters : ', grid_search_corrN.best_params_)
print('Best Score : ', grid_search_corrN.best_score_)

#Predicting the target variables
pred_corrN=grid_search_corrN.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_corrN)

Pkl_Filename_2 = "Pickle_XGBwithCorrN_Model.pkl"  

with open(Pkl_Filename_2, 'wb') as file:  
    pickle.dump(grid_search_corrN, file)

#Selected features through Mutual Information Statistics for target Hvac_S : top_40_withmut_score.index

feature_mut_S=feature.loc[:,top_40_withmut_score.index]

##We will keep 10% data for demo purpose.
X7, X_test_demo7, y7, y_test_demo7 = train_test_split(feature_mut_S,target, random_state=42, test_size=0.1)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X7,y7.hvac_S, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo7.to_csv('X_test_XGBmutS_demo.csv', index=False)  
y_test_demo7.to_csv('y_test_XGBmutS_demo.csv', index=False)  
## Saving the data for demo modelling
X7.to_csv('X_train_XGBmutS_demo.csv', index=False) 
y7.to_csv('y_train_XGBmutS_demo.csv', index=False)
# Building the model
xgb_mutS = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [9,10,12], 'learning_rate': [0.01,0.05,0.1, 0.2]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_mutS = GridSearchCV(xgb_mutS, param_grid, cv=kf)
grid_search_mutS.fit(X_train, y_train)
print('Best Parameters : ', grid_search_mutS.best_params_)
print('Best Score : ', grid_search_mutS.best_score_)

#Predicting the target variables
pred_mutS=grid_search_mutS.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mutS)

### We will be saving this model for Demo:

Pkl_Filename_3 = "Pickle_XGBwithMutS_Model.pkl"  

with open(Pkl_Filename_3, 'wb') as file:  
    pickle.dump(grid_search_mutS, file)

#Selected features through Mutual Information Statistics for target Hvac_N : top_40_withmut_score_N.index

feature_mut_N=feature.loc[:,top_40_withmut_score_N.index]

##We will keep 10% data for demo purpose.
X8, X_test_demo8, y8, y_test_demo8 = train_test_split(feature_mut_N,target, random_state=42, test_size=0.1)


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X8,y8.hvac_N, random_state=42, test_size=0.3)
# Building the model
xgb_mutN = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [6, 7, 9 , 10, 12], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_mutN = GridSearchCV(xgb_mutN, param_grid, cv=kf)
grid_search_mutN.fit(X_train, y_train)
print('Best Parameters : ', grid_search_mutN.best_params_)
print('Best Score : ', grid_search_mutN.best_score_)

#Predicting the target variables
pred_mutN=grid_search_mutN.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mutN)

feature_mut_N=feature.loc[:,top_40_withmut_score_N.index]
##We will keep 10% data for demo purpose.
X4, X_test_demo4, y4, y_test_demo4 = train_test_split(feature_mut_N,target, random_state=42, test_size=0.1)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X4,y4, random_state=42, test_size=0.3)

#Creating Polynomial feature and then performing linear regression with Robust Scalar transformaton
steps = [
    ('scalar', StandardScaler()),
  #  ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y=pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y)

#Creating Polynomial feature and then performing linear regression with Standard Scalar and Ridge Regularization
steps = [
    ('scalar', RobustScaler()),
 #   ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Ridge(alpha=0.06, fit_intercept=True))
]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)

print('Training score: {}'.format(ridge_pipeline.score(X_train, y_train)))
print('Test score: {}'.format(ridge_pipeline.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_ridge=ridge_pipeline.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_ridge)

steps = [
    ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=100, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test,y_test)))

#Predicting the target variables
pred_mut_y_lasso=lasso_pipe.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mut_y_lasso)

#------------------- End of Multiple Linear Regression Model ---------

#Selected features through Cooreleation Statistics for target Hvac_S : top_40_withcorr_score_S.index
feature_corr_S=feature.loc[:,top_40_withcorr_score_S.index]

##We will keep 10% data for demo purpose.
X5, X_test_demo5, y5, y_test_demo5 = train_test_split(feature_corr_S,target, random_state=42, test_size=0.1)
                                                      
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X5,y5.hvac_S, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo5.to_csv('X_test_XGBcorrS_demo.csv', index=False)  
y_test_demo5.to_csv('y_test_XGBcorrS_demo.csv', index=False)  
## Saving the data for demo modelling
X5.to_csv('X_train_XGBcorrS_demo.csv', index=False) 
y5.to_csv('y_train_XGBcorrS_demo.csv', index=False)
# Building the model
xgb_corrS = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=5)
grid_search_corrS = GridSearchCV(xgb_corrS, param_grid, cv=kf)
grid_search_corrS.fit(X_train, y_train)
print('Best Parameters : ', grid_search_corrS.best_params_)
print('Best Score : ', grid_search_corrS.best_score_)

#Predicting the target variables
pred_corrS=grid_search_corrS.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_corrS)


with open(Pkl_Filename_HvacS, 'wb') as file:  
    pickle.dump(grid_search_corrS, file)

#Selected features through Cooreleation Statistics for target Hvac_N : top_40_withcorr_score_N.index
feature_corr_N=feature.loc[:,top_40_withcorr_score_N.index]

##We will keep 10% data for demo purpose.
X6, X_test_demo6, y6, y_test_demo6 = train_test_split(feature_corr_N,target, random_state=42, test_size=0.1)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X6,y6.hvac_N, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo6.to_csv('X_test_corrN_demo.csv', index=False)  
y_test_demo6.to_csv('y_test_corrN_demo.csv', index=False)  
# Building the model
xgb_corrN = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_corrN = GridSearchCV(xgb_corrN, param_grid, cv=kf)
grid_search_corrN.fit(X_train, y_train)
print('Best Parameters : ', grid_search_corrN.best_params_)
print('Best Score : ', grid_search_corrN.best_score_)

#Predicting the target variables
pred_corrN=grid_search_corrN.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_corrN)


with open(Pkl_Filename_2, 'wb') as file:  
    pickle.dump(grid_search_corrN, file)

#Selected features through Mutual Information Statistics for target Hvac_S : top_40_withmut_score.index

feature_mut_S=feature.loc[:,top_40_withmut_score.index]

##We will keep 10% data for demo purpose.
X7, X_test_demo7, y7, y_test_demo7 = train_test_split(feature_mut_S,target, random_state=42, test_size=0.1)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X7,y7.hvac_S, random_state=42, test_size=0.3)
## Saving the data for Demo
X_test_demo7.to_csv('X_test_XGBmutS_demo.csv', index=False)  
y_test_demo7.to_csv('y_test_XGBmutS_demo.csv', index=False)  
## Saving the data for demo modelling
X7.to_csv('X_train_XGBmutS_demo.csv', index=False) 
y7.to_csv('y_train_XGBmutS_demo.csv', index=False)
# Building the model
xgb_mutS = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [9,10,12], 'learning_rate': [0.01,0.05,0.1, 0.2]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_mutS = GridSearchCV(xgb_mutS, param_grid, cv=kf)
grid_search_mutS.fit(X_train, y_train)
print('Best Parameters : ', grid_search_mutS.best_params_)
print('Best Score : ', grid_search_mutS.best_score_)

#Predicting the target variables
pred_mutS=grid_search_mutS.predict(X_test)

#Model Evaluation
evaluate_regression_model(y_test,pred_mutS)

### We will be saving this model for Demo:

Pkl_Filename_3 = "Pickle_XGBwithMutS_Model.pkl"  

with open(Pkl_Filename_3, 'wb') as file:  
    pickle.dump(grid_search_mutS, file)

#Selected features through Mutual Information Statistics for target Hvac_N : top_40_withmut_score_N.index

feature_mut_N=feature.loc[:,top_40_withmut_score_N.index]

##We will keep 10% data for demo purpose.
X8, X_test_demo8, y8, y_test_demo8 = train_test_split(feature_mut_N,target, random_state=42, test_size=0.1)


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X8,y8.hvac_N, random_state=42, test_size=0.3)
# Building the model
xgb_mutN = XGBRegressor(objective='reg:squarederror')
# Setting up hyperparameter grid
param_grid = {'max_depth': [6, 7, 9 , 10, 12], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}

# k-fold cross validation using GridSearch
kf = KFold(n_splits=8)
grid_search_mutN = GridSearchCV(xgb_mutN, param_grid, cv=kf)
grid_search_mutN.fit(X_train, y_train)
print('Best Parameters : ', grid_search_mutN.best_params_)
print('Best Score : ', grid_search_mutN.best_score_)

#Predicting the target variables
pred_mutN=grid_search_mutN.predict(X_test)
#Model Evaluation
evaluate_regression_model(y_test,pred_mutN)
