# # Customer/Card Detection techniques
# 
# One of the main challenges of this competition (https://www.kaggle.com/c/ieee-fraud-detection) is **how you obtain the card or the customer whose the transactions, fraudulent or not, belong to.** It has been said that the datasets have all the information to get this, but we faced two problems:
# 1. It is **anonymized**, they're collected by Vesta’s fraud protection system and digital security partners. 
# 2. The true meaning of the columns that seems to compose the card or the customer is somewhat obscure too. **The field names are masked** and pairwise dictionary will not be provided for privacy protection and contract agreement.
# 
# > For this reason, the identification of both customer or card is one of the most discussed issues in the competition these datasets come from, and the approachs and solutions are quite different.
# > **In this notebook, we are based partially on the first approach coming from one of the winners of the competition Chris Deotte teaming with Konstantin Yakovlev. We will use the same analysis techniques, but how we implement the model using this information will finally difer.**
# 
# 
# ## Our approaches
# 
# ### `uid1`
# We used a simple approach from the more evident columns: 
# * `card1`:  Probably, card number given by the issuing company or maybe customer number.
# * `addr1`:  Billing address zip code
# * `D1achr`: `D1` could be or the days passed between the transaction and the card issue date, or between transaction and the first transaction done with it. For us, this does not really matters, as can be used them both to identify the card. We made this attribute substracting `D1` (in days) to the column `TransactionDelt` (in seconds) with the formula:  
# 
# **`D1achr` = `TransactionDT` /(60x60x24) - `D1`**  
# **`uid1` = `card1` + `addr1` + `D1achr`**  
# 
# ---
# 
# ### `uid2`
# Our second approach was to add more card and address field to get the unique card or user id, similar to the choice made by Taemyung Heo (https://www.kaggle.com/c/ieee-fraud-detection/discussion/111696). When doing the FE, this uid has been really helpful, and the aggregations created with this uid improves greatly the model.
# 
# **`uid2` = `card1` + `card2` ... + `card6` + `addr1` + `addr2` + `D1achr` + `ProductCD`**
# 
# ---
# 
# ### `uid3`
# 
# Apart from these approaches, we wanted to replicate the analysis done by Chris Deotte performing adversarial validation which is based on mixing the data from train and test, removing the business target (`isFraud`) and transaction identification columns and runing a model to determine if we can predict which observations are from train and which from test, and assuming that the most important features are strogly related to the customer identification.
# 
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510
# 
# **We will use a combination of the three approaches to create new features that identifies the card and new aggregates from them.**

 


get_ipython().system('pip -q install --upgrade pip')
get_ipython().system('pip -q install --upgrade seaborn')


 


import gc
import catboost
import lightgbm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


 


##### Functions

def get_denominator(serie):
    
    array_n   = np.array(serie)
    uniques_n = np.unique(array_n[~np.isnan(array_n)])
    
    result  = None
    for i in range(1,1000):
        decimals = uniques_n*i - np.floor(uniques_n*i)

        for decimal in decimals:
            integer = []
            if ((decimal < 0.001) | (decimal > 0.999)):
                integer.append(True)
            else:
                integer.append(False)
                break

        if all(integer):
            result = i
            break

    print('denominator', serie.name, ':', result)
    
def get_fraud_weights(serie):
    
    head        = 10
    values      = list(serie.value_counts(dropna=False).index[:head])
    df_wg_fraud = pd.DataFrame(columns=[serie.name, 'count', 'weight fraud'])
    
    for v in values:
        if (v == v): # is not nan
            n = train.query('{} == "{}" & isFraud == 1'.format(serie.name, v)).shape[0]
            c = train.query('{} == "{}"'.format(serie.name, v)).shape[0]
        else:
            n = train.query('{} != {} & isFraud == 1'.format(serie.name, serie.name)).shape[0]
            c = train.query('{} != {}'.format(serie.name, serie.name)).shape[0]

        w = n/c
        df_wg_fraud = df_wg_fraud.append({serie.name:v, 'count': c, 'weight fraud':w}, ignore_index = True)                      .sort_values('weight fraud', ascending=False)
        
    return df_wg_fraud.head(head)


 


##### Download of files.

print('Downloading datasets...')
print(' ')
train = pd.read_pickle('/kaggle/input/1-ieee-cis-memory-reduction/train_mred.pkl')
print('Train has been downloaded... (1/2)')
test  = pd.read_pickle('/kaggle/input/1-ieee-cis-memory-reduction/test_mred.pkl')
print('Test has been downloaded... (2/2)')
print('Done!')


# # Brief Exploratory Data Analysis for the customer related columns
# Info and description of the columns from `TransactionID` to `R_emaildomain` and `Di` columns. This can be extended after getting the results of the adversarial validation

 


##### Info for the colums until R_emaildomain

columns = train.iloc[:,:17]
columns.info()


 


##### Main statistics

columns.describe()


 


##### Number of nulls and percentages
#dist1 greater than 90% of nulls

print('Number of NaNs in train (amount):\n', columns.isnull().sum(), sep='')
print(' ')
print('Number of NaNs in train (% of total):\n', columns.isnull().sum()/train.shape[0], sep='')


# `TransactionID` - Transactions identity information.

 


print('Is transaction unique?:', len(train.TransactionID.unique()) == train.shape[0]) # unique transaction id -> (checking that it is truly unique)


# `isFraud` - As the transactions have been reported as fraudulent, all the following transactions with the same customer account, billing address, or email address will be also labeled as fraudulent.

 


print('Fraud unbalance:', train.isFraud.value_counts()[0]/train.shape[0]) # balance legit/totals
print('Fraud correlations: \n', abs(train.corrwith(train.isFraud)).head(10).sort_values(ascending = False), sep='')


# `TransactionDT` - Timedelta from a given reference datetime (not an actual timestamp).Delta between the begining of the dataset (86400 secs) and the transaction 183 days, aprox half a year. The unit is a second. We are only using this variable to create `DeltaDays`.
# 
# `DeltaDays` = np.floor(`TransactionDT` / 60 / 60 / 24) -> converts seconds to days.

 


print('TransactionDT min:', train.TransactionDT.min()) # From the discussions, 86400 id 60*60*24, a day in seconds
print('TransactionDT max:', train.TransactionDT.max()) 
print('TransactionDT days (train):', round(train.TransactionDT.max()/train.TransactionDT.min())) # Days in 6 months in train
print('TransactionDT days (test):', round((test.TransactionDT.max()-test.TransactionDT.min())/train.TransactionDT.min())) # Days in 6 months in train


# `TransactionAmt` - Transaction amount.

 


print('TransactionAmt min:', train.TransactionAmt.min())
print('TransactionAmt max:', train.TransactionAmt.max())
print('TransactionAmt mean:', train.TransactionAmt.mean())
print('TransactionAmt correlations: \n', abs(train.corrwith(train.TransactionAmt)).sort_values(ascending = False).head(10), sep='')
print('TransactionAmt fraud weights: \n', get_fraud_weights(train.TransactionAmt), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'TransactionAmt', hue='isFraud', bins=40)
plt.show()


# `ProductCD` - 5 different product code, the product for each transaction.

 


print('ProductCD fraud weights: \n', get_fraud_weights(train.ProductCD), sep='')


 


plt.figure(figsize=(12,6))
sns.countplot(x = train.ProductCD.values, hue= train.isFraud)
plt.show()


# `card1` - Probably, this is the number or part of the number given by the issuing entity. Categorical. It is not ordinal.

 


print('card1 values counts: \n', train.card1.value_counts(dropna=False), sep='')
print('card1 fraud weights: \n', get_fraud_weights(train.card1))
print('card1 correlations: \n', abs(train.corrwith(train.card1)).sort_values(ascending = False).head(10))


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'card1', hue='isFraud')
plt.show()


# `card2` - Categorical. Numeric, not ordinal.

 


print('card2 value counts: \n', train.card2.value_counts(dropna=False).head(10), sep='')
print('card2 fraud weights: \n', get_fraud_weights(train.card2), sep='')
print('card2 correlations: \n', abs(train.corrwith(train.card2)).sort_values(ascending = False).head(10), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'card2', hue='isFraud')
plt.show()


# `card3` - Categorical.

 


print('card3 value counts: \n', train.card3.value_counts(dropna=False).head(10), sep='')
print('card3 fraud weights: \n', get_fraud_weights(train.card3), sep='')
print('card3 correlations: \n', abs(train.corrwith(train.card3)).sort_values(ascending = False).head(10), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'card3', hue='isFraud')
plt.show()


# `card4` - Card payments company. Categorical, strings.

 


print('card4 value counts: \n', train.card4.value_counts(dropna=False).head(10), sep='')
print('card4 fraud weights: \n', get_fraud_weights(train.card4), sep='')


 


plt.figure(figsize=(12,6))
sns.countplot(data=train, x = 'card4', hue='isFraud')
plt.show()


# `card5` - Numeric, categorical.

 


print('card5 value counts: \n', train.card5.value_counts(dropna=False).head(10), sep='')
print('card5 fraud weights: \n', get_fraud_weights(train.card5), sep='')
print('card5 correlations: \n', abs(train.corrwith(train.card5)).sort_values(ascending = False).head(10), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'card5', hue='isFraud')
plt.show()


# `card6` - Card type. Categorical, strings.

 


print('card6 value counts: \n', train.card6.value_counts(dropna=False).head(10), sep='')
print('card6 fraud weights: \n', get_fraud_weights(train.card6), sep='')


 


plt.figure(figsize=(12,6))
sns.countplot(data=train, x = 'card6', hue='isFraud')
plt.show()


# `addr1` - This is the billing region or zip code. It is used to identify unique cards or user ids. In the real world, the zip codes are different between countries, but in the dataset, there are not many different ones: that depends on how the anonymization was done. This information has been analyzed in previous notebook, and it is part of the identification of a card.

 


print('addr1 value counts: \n', train.addr1.value_counts(dropna=False).head(10), sep='')
print('addr1 fraud weights: \n', get_fraud_weights(train.addr1), sep='') # NaN values have 11,7 % fraud vs 3.5%
print('addr1 correlations: \n', abs(train.corrwith(train.addr1).sort_values(ascending = False)).head(15), sep='') # Hight correlation


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'addr1', hue='isFraud')
plt.show()


# `addr2` - This is the billing country.

 


print('addr2 value counts: \n', train.addr2.value_counts(dropna=False).head(10), sep='')
print('addr2 fraud weights: \n', get_fraud_weights(train.addr2), sep='') # Country 65 has ~50% of fraud!!
print('addr2 correlations: \n', abs(train.corrwith(train.addr2)).sort_values(ascending = False).head(15), sep='') # High correlation with card3


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'addr2', hue='isFraud')
plt.show()


 


##### Inspecting addr1 and addr2 Nans
# When addr1 is null, then addr2 is null too (and the other way around). Try filling NaNs with -999.

print('addr1 null if addr not null (train):', train.query('addr1!=addr1 & addr2==addr2').shape)
print('addr2 null if add2 not null (train):', train.query('addr2!=addr2 & addr1==addr1').shape)
print('addr1 null if addr not null (test):' , test.query('addr1!=addr1 & addr2==addr2').shape)
print('addr2 null if add2 not null (test):' , test.query('addr2!=addr2 & addr1==addr1').shape)


# `P_emaildomain` - This is the mail domain of the purchaser.

 


# NaNs to be filled with the mode of the card, the rest with unknown and then to -999

print('P_emaildomain value counts: \n', train.P_emaildomain.value_counts(dropna=False).head(10), sep='')
print('P_emaildomain fraud weights: \n:', get_fraud_weights(train.P_emaildomain), sep='')


 


plt.figure(figsize=(12,6))
sns.countplot(data=train, x = 'P_emaildomain', hue='isFraud')
plt.xticks(rotation=90)
plt.show()


# `R_emaildomain` - This is the mail domain of the recipient.

 


# NaNs to be filled with the mode of the card, the rest with unknown and then to -999

print('R_emaildomain value counts: \n', train.R_emaildomain.value_counts(dropna=False).head(10), sep='')
print('R_emaildomain fraud weights: \n:', get_fraud_weights(train.R_emaildomain), sep='')


 


plt.figure(figsize=(12,6))
sns.countplot(data=train, x = 'R_emaildomain', hue='isFraud')
plt.xticks(rotation=90)
plt.show()


# `dist1` - Distance between, but not limited to, billing address, delivery address, telephonic area, etc.

 


print('dist1 value counts: \n', train.dist1.value_counts(dropna=False).head(10), sep='')
print('dist1 fraud weights: \n', get_fraud_weights(train.dist1), sep='')
print('dist1 correlations: \n', train.corrwith(train.dist1).sort_values(ascending = False).head(15), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'dist1', hue='isFraud', bins=35)
plt.show()


# `dist2` - Distance between, but not limited to, billing address, delivery address, telephonic area, etc.

 


print('dist2 value counts: \n', train.dist2.value_counts(dropna=False).head(10), sep='')
print('dist2 fraud weights: \n', get_fraud_weights(train.dist2), sep='')
print('dist2 correlations: \n', train.corrwith(train.dist2).sort_values(ascending = False).head(15), sep='')


 


plt.figure(figsize=(12,6))
sns.histplot(data=train, x = 'dist2', hue='isFraud', bins=35)
plt.show()


# `Di columns`

 


# All are in days, except D8 and probably D9. D8, D9 look they are in hours, not in days.

for col in list(train.iloc[:,31:46]):
    print(col, ':', len(train[col].value_counts()))
    
print('#' * 50)    
get_denominator(train['D8'])
get_denominator(train['D9'])


 


# D6, D7, D8, D9, D12, D13, D14 about 90% nulls

print(train.iloc[:,31:46].describe())
print()
print('D null count: \n', train.iloc[:,31:46].isnull().sum()/train.shape[0], sep='')


 


plt.figure(figsize=(10,10))
sns.heatmap(train.iloc[:,31:46].corr(), annot=True, fmt='.2g', cmap='Reds')
plt.show()


 


#D3 days from the previous card transaction, D5 and D7 looks something similar. D3 and D5 quite similar. Probably we will remove D7, but due to it gets 93% of NaNs.

train.iloc[:,31:46].query('D3==D3 & D5==D5 & D7==D7 & D5!=D7')[['D3', 'D5', 'D7']].head(10)


# # Adversarial Validation

# ## Analysis of the first 54 columns

 


##### Adding the Dxachr columns to perform the validation instead of de original Ds. These columns already come from the binaries loaded.

train_til_M = train.iloc[:,:55].copy()
train_til_M = train_til_M.drop('isFraud', axis=1)
train_til_M['is_train'] = 1

train_til_M = pd.concat([train_til_M, train.iloc[:,-10:]], axis=1)

test_til_M = test.iloc[:,:54].copy()
test_til_M['is_train'] = 0
test_til_M = pd.concat([test_til_M, test.iloc[:,-10:]], axis=1)

train_test_til_M = pd.concat([train_til_M, test_til_M], axis=0, ignore_index=True)

del train_til_M
del test_til_M

gc.collect()


 


##### Basic Fill NaNs for LGBM: Get categories and preprocess with Label Encoder.

categ_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
categ_cols += ['M'+str(i) for i in range(1,10)]

# Replace null by not_know
train_test_til_M.loc[:,categ_cols] = train_test_til_M[categ_cols].fillna('not_know')

# Label Encoder
le = LabelEncoder()
for col in categ_cols:
    le.fit(train_test_til_M[col])
    train_test_til_M[col] = le.transform(train_test_til_M[col])


 


##### Adversarial Validation - sample

adversarial_val   = train_test_til_M.sample(200000, replace=False)
adversarial_train = train_test_til_M[~train_test_til_M.index.isin(adversarial_val.index)]


 


##### Remove columns that we know relates to transaction and keep the rest.

features = list(train_test_til_M)
features.remove('TransactionID') 
features.remove('TransactionDT')
features.remove('is_train')

# Remove the original Dxs for those we have the engineered Dxahcrs
columns_D = ['D' + str(i) for i in range(1,16) if i not in [3,5,7,8,9]]
for col in columns_D: 
    features.remove(col)
    
# Target definition
target = 'is_train'

del train_test_til_M


 


##### Adversarial validation with LGBM

train_data    = lightgbm.Dataset(adversarial_train[features], label=adversarial_train[target], categorical_feature=categ_cols)
test_data     = lightgbm.Dataset(adversarial_val[features], label=adversarial_val[target], categorical_feature=categ_cols)

parameters = {
                    'objective':'binary', # in classification, binary or multiclass
                    'boosting_type':'gbdt', # boosting type, gradient boosting decission trees, dart, goss. Behind XGBoost
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.5,      #each tree will use this percentage of rows (randomly). Smaller, improves generalization and speed. Not too low, as we look for overfiting.
                    'n_estimators':1000,  #number of trees. This mean that there will be N trees, that they will be executed one after another, unless early_stopping_rounds condition is met.
                    'max_bin':255,
                    'verbose':-1,
                    'early_stopping_rounds':100, 
                } 

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data, verbose_eval=200)


 


##### AUC

print('AUC:',model.best_score.get('valid_0').get('auc'))


 


##### Feature Importance

features_importance = model.feature_importance()
features_array = np.array(features)
features_array_ordered_i = features_array[(features_importance).argsort()[::-1]]

plt.figure(figsize=(16,10))
sns.barplot(y=features, x=features_importance, orient='h', order=features_array_ordered_i)
plt.show()

del model


# ## Analysis of V columns

 


train_V = train.iloc[:,55:-11:].copy()
train_V['is_train'] = 1


test_V = test.iloc[:,55:-11:].copy()
test_V['is_train'] = 0

train_test_V = pd.concat([train_V, test_V], axis=0, ignore_index=True)

del train_V
del test_V
gc.collect()


 


##### Adversarial Validation - sample

adversarial_val   = train_test_V.sample(200000, replace=False)
adversarial_train = train_test_V[~train_test_V.index.isin(adversarial_val.index)]

del train_test_V

# Remove columns that we know relates to transaction and keep the rest

features = list(adversarial_train)
features.remove('is_train')

# Target definition

target = 'is_train'


 


##### Adversarial validation with LGBM

train_data    = lightgbm.Dataset(adversarial_train[features], label=adversarial_train[target])
test_data     = lightgbm.Dataset(adversarial_val[features],   label=adversarial_val[target])

parameters = {
                    'objective':'binary', # in classification, binary or multiclass
                    'boosting_type':'gbdt', # boosting type, gradient boosting decission trees, dart, goss. Behind XGBoost
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.5,      #each tree will use this percentage of rows (randomly). Smaller, improves generalization and speed. Not too low, as we look for overfiting.
                    'n_estimators':1000,  #number of trees. This mean that there will be N trees, that they will be executed one after another, unless early_stopping_rounds condition is met.
                    'max_bin':255,
                    'verbose':-1,
                    'early_stopping_rounds':100, 
                } 

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data, verbose_eval=200)


 


##### AUC

print('AUC:',model.best_score.get('valid_0').get('auc'))


 


##### Feature Importance

features_importance = model.feature_importance()
features_array = np.array(features)
features_array_ordered_v = features_array[(features_importance).argsort()[::-1]]

plt.figure(figsize=(16,10))
sns.barplot(y=features, x=features_importance, orient='h', order=features_array_ordered_v[:30])
plt.show()


# ## Finally, we train the model for the adversarial validation with the best 50% of the first batch and the the best 50% of the second

 


mixed_features_array = np.hstack([
    features_array_ordered_i[:int(np.floor(len(features_array_ordered_i)/2))],
    features_array_ordered_v[:int(np.floor(len(features_array_ordered_v)/2))]])


 


train_M = train[mixed_features_array].copy()
train_M['is_train'] = 1

test_M = test[mixed_features_array].copy()
test_M['is_train'] = 0

train_test_M = pd.concat([train_M, test_M], axis=0, ignore_index=True)

del train_M
del test_M
del train
del test
gc.collect()


 


categ_cols = list(train_test_M.dtypes[train_test_M.dtypes == 'O'].index)

train_test_M.loc[:,categ_cols] = train_test_M[categ_cols].fillna('not_know')

le = LabelEncoder()
for col in categ_cols:
    le.fit(train_test_M[col])
    train_test_M[col] = le.transform(train_test_M[col])


 


##### Adversarial Validation - sample

adversarial_val   = train_test_M.sample(200000, replace=False)
adversarial_train = train_test_M[~train_test_M.index.isin(adversarial_val.index)]

del train_test_M

# Remove columns that we know relates to transaction and keep the rest
features = list(adversarial_train)
features.remove('is_train')

for col in columns_D: # Remove the original Dxs for those we have the engineered Dxahcrs
    try:
        features.remove(col)
    except:
        pass
    
# Target definition
target = 'is_train'


 


##### Adversarial validation with LGBM

train_data    = lightgbm.Dataset(adversarial_train[features], label=adversarial_train[target], categorical_feature=categ_cols)
test_data     = lightgbm.Dataset(adversarial_val[features],   label=adversarial_val[target],   categorical_feature=categ_cols)

parameters = {
                    'objective':'binary', # in classification, binary or multiclass
                    'boosting_type':'gbdt', # boosting type, gradient boosting decission trees, dart, goss. Behind XGBoost
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.5,      #each tree will use this percentage of rows (randomly). Smaller, improves generalization and speed. Not too low, as we look for overfiting.
                    'n_estimators':1000,  #number of trees. This mean that there will be N trees, that they will be executed one after another, unless early_stopping_rounds condition is met.
                    'max_bin':255,
                    'verbose':-1,
                    'early_stopping_rounds':100, 
                } 

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data, verbose_eval=200)


 


##### AUC

print('AUC:',model.best_score.get('valid_0').get('auc'))


 


##### Feature Importance

features_importance = model.feature_importance()
features_array = np.array(features)
features_array_ordered_v = features_array[(features_importance).argsort()[::-1]]

plt.figure(figsize=(16,10))
sns.barplot(y=features, x=features_importance, orient='h', order=features_array_ordered_v[:30])
plt.show()


# # Conclusions
# 
# Attending the adversarial validation, these characteristics can be strongly related to the customer identification:
# 
# * `card1` y/o `card2`
# * `D1achr` y/o `D11achr`, `D10achr`, `D15achr`, `D4achr`
# * `C13`
# * `addr1`
# * `dist1`
# 
# We tested the model outcomes, engineering the features based on this uid, f.e. creating aggregations, against our original (card columns +addr columns + ProductCD + D1achr) and the results were better with the latter than the former. However, these features are also important for the fraud detection model. We are using a mix between these two and the first uid approach in order to obtain new features that help the model to differenciate between fraudulent and legit transactions.
