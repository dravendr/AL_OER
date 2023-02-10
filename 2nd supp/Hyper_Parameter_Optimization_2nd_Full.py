###########import packages##########
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
###########import packages##########
import catboost
import xgboost
import lightgbm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import *
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  *
###########import packages##########
import tensorflow as tf
import keras
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.constraints import max_norm
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.models import Model
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm 
# from keras.wrappers import scikit_learn
from scikeras.wrappers import KerasClassifier, KerasRegressor
###########loading data##########
loo = LeaveOneOut()
# %matplotlib
###########wrapping root mean square error for later calls##########
def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE=np.sqrt(sum(squaredError)/len(squaredError))
    R2=r2_score(target,prediction)
    return mae,mse,RMSE,R2

def gridsearch(model,param,algorithm_name,X,y):
#     grid = GridSearchCV(model,param_grid=param,scoring='neg_mean_absolute_error',cv=10,n_jobs=-1,verbose=2)
    grid = GridSearchCV(model,param_grid=param,scoring='neg_mean_absolute_error',cv=10,n_jobs=-1,verbose=-1)
    grid.fit(X,y)
    best_model=grid.best_estimator_
    
    prediction_all = best_model.predict(X)
    real_all=y.values
    prediction_all_series=pd.Series(prediction_all)
    real_all_series=pd.Series(real_all)
    
    ###########evaluating the regression quality##########
    corr_ann = round(prediction_all_series.corr(real_all_series), 5)
    error_val= compute_mae_mse_rmse(prediction_all,real_all)
    
    print(algorithm_name)
    best_score=grid.best_score_
    print('Best Regressor:',grid.best_params_,'Best Score:', best_score)

    return best_model,best_score

fl = open(r'./database_full_ac.pkl','rb')
database_full=pickle.load(fl)
data_input_full=database_full.iloc[:,0:54]
data_output_full=database_full.iloc[:,54]

model_SVR = svm.SVR()
param_svr = {
     'kernel':['linear', 'poly', 'rbf'],
              'max_iter':[100,200,300,400,500,600,700,800,1000,1100,1200,1300,1400,1500],
          'degree':[2,3,4],
         'gamma':['scale','auto'],
          'epsilon':[0.001,0.01,0.1,0.3,0.5,0.7,1],
              'coef0':[100,200,300,400,500,600,700,800,1000,1100,1200,1300,1400,1500]
       }
SVR_full,SVR_full_score=gridsearch(model_SVR,param_svr,'Support Vector Regressor',data_input_full,data_output_full)

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
param_knr = {
    'n_neighbors':range(1,10),'weights':['uniform','distance'],
             'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
         'leaf_size':[2,10,20,30,40,50,100],
         'p':range(1,10)
       }
KNR_full,KNR_full_score=gridsearch(model_KNeighborsRegressor,param_knr,'K Nearest Neighbor Regressor',data_input_full,data_output_full)

model_LGBMRegressor=LGBMRegressor(random_state=1,verbose=0)
param_lgbm = {
'boosting_type':['gbdt','rf'],
'learning_rate':[0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.04,0.06,0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.16,0.18,0.2,0.4,0.5,0.6,0.8,1],
'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'n_estimators':[50,100,200,400],
'max_depth':[5,7,9,11,13,-1],
'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
LGBM_full,LGBM_full_score=gridsearch(model_LGBMRegressor,param_lgbm,'LightGBM',data_input_full,data_output_full)

model_XGRegressor=XGBRegressor(random_state=1)
param_xg={
'booster':['gbtree'],
'learning_rate':[0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.04,0.06,0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.16,0.18,0.2,0.4,0.5,0.6,0.8,1],
'n_estimators':[100,200,400],
'max_depth':[3,5,7,9,11,13,-1],
'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
XG_full,XG_full_score=gridsearch(model_XGRegressor,param_xg,'XGBoost',data_input_full,data_output_full)

model_CatRegressor=catboost.CatBoostRegressor(random_state=1,verbose=0)
param_cat = {
'learning_rate':[0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.04,0.06,0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.16,0.18,0.2],
'n_estimators':[100,200,400],
"boosting_type":["Plain"],
'max_depth':[5,7,9,11],
'subsample':[0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
CAT_full,CAT_full_score=gridsearch(model_CatRegressor,param_cat,'CatBoost',data_input_full,data_output_full)

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(random_state=1)
###########defining the parameters dictionary##########
param_GB = {
'learning_rate':[0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.04,0.06,0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.16,0.18,0.2,0.4,0.5,0.6,0.8,1],
'n_estimators':[50,100,200,400],
'max_depth':[3,5,7,9,11,13,16],
'criterion':['friedman_mse','mae','mse'],
'max_features':['auto','sqrt','log2'],
'loss':['ls', 'lad', 'huber', 'quantile']
}
GB_full,GB_full_score=gridsearch(model_GradientBoostingRegressor,param_GB,'GradientBoost',data_input_full,data_output_full)

###########RandomForest gridsearch CV for best hyperparameter##########
model_RandomForestRegressor = ensemble.RandomForestRegressor(random_state=1)
###########defining the parameters dictionary##########
param_RF = {
'n_estimators':[50,100,200,400,None],
'max_depth':[3,5,7,9,11,None],
'criterion':['mse','mae'],
'max_features':['auto','sqrt','log2']
}
RF_full,RF_full_score=gridsearch(model_RandomForestRegressor,param_RF,'Random Forest',data_input_full,data_output_full)

model_DecisionTreeRegressor = tree.DecisionTreeRegressor(random_state=1)
param_dt={
'max_depth':[5,6,7,8,9,10,11,None],
'max_features':['auto','sqrt','log2'],
'criterion' : ["mse", "friedman_mse", "mae"],
'splitter' : [ "best",'random']
}
DT_full,DT_full_score=gridsearch(model_DecisionTreeRegressor,param_dt,'Decision Tree',data_input_full,data_output_full)

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(random_state=1)
param_ada={
'n_estimators':[50,100,200,400,800],
'learning_rate':[0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.04,0.06,0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.16,0.18,0.2,0.4,0.5,0.6,0.8,1],
'loss':['linear', 'square', 'exponential'] 
}
ADA_full,ADA_full_score=gridsearch(model_AdaBoostRegressor,param_ada,'AdaBoost',data_input_full,data_output_full)

def create_ANN_model_1layer(X,learning_rate,regular_term=0.001,neuron_number=50,drop_out_rate=0):
    regularizer=keras.regularizers.l2(regular_term)
    model = Sequential() 
    model.add(Dense(neuron_number, input_dim=X.shape[1], kernel_initializer='random_normal',
                    bias_initializer='random_normal',activation='relu',kernel_regularizer=regularizer)) 
    model.add(Dropout(drop_out_rate))
    model.add(Dense(neuron_number, input_dim=neuron_number, kernel_initializer='random_normal',
                    bias_initializer='random_normal',activation='relu',kernel_regularizer=regularizer)) 
    model.add(Dropout(drop_out_rate))
    model.add(Dense(1, input_dim=neuron_number, activation='linear'))
    adam=optimizers.Adam(learning_rate)
    model.compile(loss='mae')
    return model
def create_ANN_model_2layer(X,learning_rate,regular_term=0.001,neuron_number=50,drop_out_rate=0):
    regularizer=keras.regularizers.l2(regular_term)
    model = Sequential() 
    model.add(Dense(neuron_number, input_dim=X.shape[1], kernel_initializer='random_normal',
                    bias_initializer='random_normal',activation='relu',kernel_regularizer=regularizer)) 
    model.add(Dropout(drop_out_rate))
    model.add(Dense(neuron_number, input_dim=neuron_number, kernel_initializer='random_normal',
                    bias_initializer='random_normal',activation='relu',kernel_regularizer=regularizer)) 
    model.add(Dropout(drop_out_rate))
    model.add(Dense(neuron_number, input_dim=neuron_number, kernel_initializer='random_normal',
                    bias_initializer='random_normal',activation='relu',kernel_regularizer=regularizer)) 
    model.add(Dropout(drop_out_rate))
    model.add(Dense(1, input_dim=neuron_number, activation='linear'))
    adam=optimizers.Adam(learning_rate)
    model.compile(loss='mae')
    return model

model_ANNRegressor1= KerasRegressor(build_fn=create_ANN_model_1layer(X=data_input_full,learning_rate=0.01), verbose=0)
model_ANNRegressor2= KerasRegressor(build_fn=create_ANN_model_2layer(X=data_input_full,learning_rate=0.01), verbose=0)

epochs_list=[]
for i in range(10,210,10):
    epochs_list.append(i)

# 设置参数候选值
batch_size_list = [8,16,32]
optimizers_list=['sgd', 'rmsprop', 'adam', 'adagrad']
param_ann = dict(batch_size=batch_size_list, 
                 epochs=epochs_list,
                optimizer=optimizers_list
                )

ANN_1layer,ANN_1layer_score=gridsearch(model_ANNRegressor1,param_ann,'Artificial Neural Network',data_input_full,data_output_full)
ANN_2layer,ANN_2layer_score=gridsearch(model_ANNRegressor2,param_ann,'Artificial Neural Network',data_input_full,data_output_full)
