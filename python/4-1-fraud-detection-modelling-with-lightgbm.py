import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import joblib

from bayes_opt import BayesianOptimization
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings("ignore")





##### Download of files.

print('Downloading datasets...')
print(' ')
df_train = pd.read_pickle('/kaggle/input/3-fraud-detection-preprocessing/train.pkl')
print('Train has been downloaded... (1/2)')
df_test = pd.read_pickle('/kaggle/input/3-fraud-detection-preprocessing/test.pkl')
print('Test has been downloaded... (2/2)')
print(' ')
print('All files are downloaded')





##### Feature selection for the model
# To prevent overfitting, we choose the elimination of some columns.

features = list(df_train)
features_to_remove = ['TransactionID','isFraud','TransactionDT','DT','uid1', 'uid2', 'uids', 'D1','D2','D4','D6','D10','D11','D12',
                     'D13','D14','D15']

for col in features_to_remove:
    features.remove(col)

# Get the feature list

print(f'Columns selected: {features}')
print(f'Lenght: {len(features)}')





##### Train + val separation
# We separate train (6 months) into train (4 months and 15 days) + skip (15 days) + val (1 month)

train = df_train.query('DT <= 135').copy()
val = df_train.query('DT >  150').copy()





##### Defining X and target

# Train
X_train = train[features]
y_train = train.isFraud
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

# Val
X_val = val[features]
y_val = val.isFraud
print('X_val:', X_val.shape)
print('y_val:', y_val.shape)

# Test
X_test = df_test[features]
print('X_test:', X_test.shape)


# # Modelling with LightGBM




##### Bayesian Optimization
# https://github.com/fmfn/BayesianOptimization

def objective(learning_rate,num_leaves,max_depth,colsample_bytree,subsample):
    
    # get X_train and X_val that are defined outside the functions that's why 'global' label
    global X_train
    global y_train
    global X_val
    global y_val
    
    # transform dtype of integers to avoid errors
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    
    # check dtype if not: stop and execute an error
    assert type(num_leaves) == int
    assert type(max_depth) == int
    
    # create the params
    parameters = {'objective': 'binary',
             'boosting_type': 'gbdt',
             'metric': 'auc',
             'n_jobs': -1,
             'tree_learner': 'serial',
             'learning_rate': learning_rate,
             'num_leaves': num_leaves,
             'max_depth': max_depth,
             'colsample_bytree': colsample_bytree,
             'subsample': subsample,
             'verbosity': -1,
             'random_state': 27}
    
    # create the dataset
    categorical_columns = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'M1', 'M3', 'M4', 'M5', 'M6', 'M8', 'M9']
    
    train_data = lgb.Dataset(X_train, label = y_train, categorical_feature = categorical_columns)
    val_data = lgb.Dataset(X_val, label = y_val, categorical_feature = categorical_columns)
    
    # train the model
    model_lgb = lgb.train(parameters, train_data , valid_sets = [train_data, val_data], num_boost_round = 200, categorical_feature = categorical_columns,
                          early_stopping_rounds = 200, verbose_eval = 0)
    
    score  = roc_auc_score(y_val, model_lgb.predict(X_val))
                          
    return score





params = {'learning_rate': (0.02,0.03),
         'num_leaves': (225,275),
         'max_depth': (12,16),
         'colsample_bytree': (0.3,0.9),
         'subsample': (0.3,0.9)}

clf_BO = BayesianOptimization(objective, params, random_state = 27)

clf_BO.maximize(init_points = 3, n_iter = 20)





print(f'Best Parameters: {clf_BO.max["params"]}')
print(' ')
print('Best AUC:',{clf_BO.max['target']})


# |     #     | learning_rate |    num_leaves   |    max_depth    |colsample_bytree |   subsample     |       AUC       |
# |:----------:|:--------------:|:----------------:|:----------------:|:----------------:|:----------------:|----------------:|
# | 1         |0.03215518473989215|             294 |             14 |             0.4 |             0.8 |0.9194457912401148|
# | 2         |            0.04 |             278 |             13 |             0.4 |             0.9 |0.9164233793906151|
# | 3         |0.027832622205538907|             268 |             15 |0.3740990054742914|0.589171882228726|0.9188929389573968|
# | 4         |0.029827755272285032|             262 |             15 |0.7823490596970961|0.8631490082348747|0.92719808846762|




##### LightGBM with #3

parameters = {'objective': 'binary',
             'boosting_type': 'gbdt',
             'metric': 'auc',
             'n_jobs': -1,
             'tree_learner': 'serial',
             'learning_rate': 0.027832622205538907,
             'num_leaves': 268,
             'max_depth': 15,
             'colsample_bytree': 0.3740990054742914,
             'subsample': 0.589171882228726,
             'verbosity': -1,
             'random_state': 27}

categorical_columns = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'M1', 'M3', 'M4', 'M5', 'M6', 'M8', 'M9']

train_data = lgb.Dataset(X_train, label = y_train, categorical_feature = categorical_columns)
val_data   = lgb.Dataset(X_val,   label = y_val, categorical_feature = categorical_columns)

clf = lgb.train(parameters, train_data , valid_sets = [train_data, val_data], num_boost_round = 200, categorical_feature = categorical_columns,
                          early_stopping_rounds = 200, verbose_eval = 100)





##### AUC

auc = roc_auc_score(y_val, clf.predict(X_val))
print('AUC:',auc)





##### ROC Curve

y_pred = clf.predict(X_val)
r_curve = roc_curve(y_val, y_pred)

plt.figure(figsize=(8,8))
plt.title('ROC')
plt.plot(r_curve[0], r_curve[1], 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





##### Feature Importance

features_importance = clf.feature_importance()
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y=features_array, x=features_importance, orient='h', order=features_array_ordered[:50])

plt.show()





##### Saving this results into an Excel file

try:
    model_results = pd.read_excel('model_results.xlsx')
    
    model_results = model_results.append({'datetime' : datetime.datetime.now(), 'clf':clf.__class__, 'features': features_array, 'parameters' : clf.params,
                                       'AUC': auc, 'features_importance': list(features_importance), 'features_ordered': list(features_array_ordered), 
                                       }, ignore_index = True)
        
except:
    model_results = pd.DataFrame(columns = ['summary', 'datetime', 'clf', 'features', 'parameters', 'entry1', 'entry2', 
                                            'AUC', 'features_importance', 'features_ordered', 'output1', 'output2'])
    
    model_results = model_results.append({'datetime' : datetime.datetime.now(), 'clf':clf.__class__, 'features': features_array, 'parameters' : clf.params, 
                                       'AUC': auc, 'features_importance': list(features_importance), 'features_ordered': list(features_array_ordered),
                                       }, ignore_index = True)
finally:
    model_results.to_excel('model_results.xlsx', index= False)





##### Saving this model

joblib.dump(clf, 'model_lightgbm.pkl')


# # Submission to Kaggle




pred = clf.predict(X_test)
submission = pd.DataFrame()

submission['TransactionID'] = df_test['TransactionID']
submission['isFraud'] = pred

submission
submission.to_csv('submission.csv', index = False)





val.to_pickle('val_lgb.pkl')
print('Done!')

