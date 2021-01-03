import lightgbm

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd 

from memory_reduction_script import reduce_mem_usage_sd as mr # We are using the memory reduction by our utility script





##### Functions
# 1st function

def j_mode(x, dropna=True):
    #obtains the most frequent, not nan value
    try: 
        mode = x.value_counts(dropna=dropna).index[0]
        # mode != mode -> isnull
        if (mode != mode) and (x.value_counts(dropna=dropna).index > 1):
            mode = x.value_counts(dropna=dropna).index[1]            
        return mode
    except: 
        return x

# 2nd function

def normalize_columns(group, cols, df_train, df_test, verbose=True):
    # replacing the card values depending on card1 with the most frequent, not nan value
    ### initialize trace variables
    if verbose:
        s_train_before    = df_train.shape
        s_train_na_before = df_train[cols].isnull().sum()
        s_test_before     = df_test.shape
        s_test_na_before  = df_test[cols].isnull().sum()

        #train_nulls_before = pd.concat([df_train[cols].isnull().sum(), df_train[cols].isnull().sum()], axis=1)
        #test_nulls_before  = pd.concat([df_test[cols].isnull().sum(), df_test[cols].isnull().sum()], axis=1)

    
    #normalize the group with the mode.
    grouped_train = df_train.groupby([group])
    grouped_test  = df_test.groupby([group])

    n = 0
    for col in cols:
        if verbose:
            print('normalizing ' + str(col))
        df_train[col] = grouped_train[col].transform(lambda x: j_mode(x))
        df_test[col]  = grouped_test[col].transform(lambda x: j_mode(x))
        n += 1
        
    ### print the traces
    if verbose:        
        print(f'train shape before: {s_train_before}, after: {df_train.shape}')
        print(f'test shape before: {s_test_before}, after: {df_test.shape}')
        train_nulls_after = df_train[cols].isnull().sum()
        test_nulls_after  = df_test[cols].isnull().sum()
        for i in range(s_test_na_before.shape[0]):
            print(f'(train) {s_train_na_before.index[i]} nulls before: {s_train_na_before.iloc[i]}, nulls after: {train_nulls_after.iloc[i]}')
        for i in range(s_test_na_before.shape[0]):
            print(f'(test) {s_test_na_before.index[i]} nulls before: {s_test_na_before.iloc[i]}, nulls after: {test_nulls_after.iloc[i]}')

# 3rd function

def fill_na(cols, df_train, df_test, num_rep=-999, obj_rep ='Unknown', verbose=True):
    ### initialize trace variables
    if verbose:
        s_train_na_before = train[cols].isnull().sum()
        s_test_na_before  = test[cols].isnull().sum()
        
    for col in cols:
        if df_train[col].dtype == 'O':
            df_train[col] = df_train[col].fillna(obj_rep)
            df_test[col]  = df_test[col].fillna(obj_rep)
        else:
            df_train[col] = df_train[col].fillna(num_rep)
            df_test[col]  = df_test[col].fillna(num_rep)

    ### print the traces
    if verbose:        
        train_nulls_after = df_train[cols].isnull().sum()
        test_nulls_after  = df_test[cols].isnull().sum()
        
        for i in range(s_test_na_before.shape[0]):
            print(f'(train) {s_train_na_before.index[i]} nulls before: {s_train_na_before.iloc[i]}, nulls after: {train_nulls_after.iloc[i]}')
        for i in range(s_test_na_before.shape[0]):
            print(f'(test) {s_test_na_before.index[i]} nulls before: {s_test_na_before.iloc[i]}, nulls after: {test_nulls_after.iloc[i]}')





##### Download of files.

print('Downloading datasets...')
print(' ')
train = pd.read_pickle('/kaggle/input/1-fraud-detection-memory-reduction/train_mred.pkl')
print('Train has been downloaded... (1/2)')
test = pd.read_pickle('/kaggle/input/1-fraud-detection-memory-reduction/test_mred.pkl')
print('Test has been downloaded... (2/2)')
print(' ')
print('All files are downloaded')





print(train.shape) # 6 months of data in train
print(test.shape)  # 6 months of data in test


# # Preprocessing:
# 
# 1. Three different card keys (ways to group cards or users):
#    
#     * `uid1` with `card1` to `card6` + `addr1` + `addr2` + `ProductCD` + `D1achr`
#     * `uid2` with `card1` + `D2achr` + `C13` + `D11achr` + `D10achr` + `D15achr` + `D4achr`
#     * `uids` with `card1` + `addr1` + `D1achr`  
#   
#   
# 2. Normalize all the fields related to card1 -> replace less frequent, NaNs values with the mode.
# 3. Input all categorical data - we encode them with a dictionary to set the same value for all NaNs.




object_columns = train.dtypes[train.dtypes=='O'].index
for col in object_columns:
    print(col ,'nulls (train):', train[col].isnull().sum()/train.shape[0])
    print(col ,'nulls (test):', test[col].isnull().sum()/test.shape[0])





##### Group outliers for card4 and card6
print(" BEFORE ".center(20, '#'))
print('Train - card6:\n', train['card6'].value_counts(dropna=False))
print(' ')
print('Train - card4:\n', train['card4'].value_counts(dropna=False))
print(' ')

train.card6 = train.card6.replace(['debit or credit', 'charge card'], np.nan)
train.card4 = train.card4.replace(['american express', 'discover'], np.nan)

print(" AFTER ".center(20, '#'))
print('Train - card6:\n', train['card6'].value_counts(dropna=False))
print(' ')
print('Train - card4:\n', train['card4'].value_counts(dropna=False))
print(' ')





##### Label encoding all object columns with our dictionary
for col in object_columns:
    print(f'String values from {col} are being transformed to numeric...')
    #unique values without nans
    unique_values = list(train[col].dropna().unique()) 

    #create the dictionary
    str_to_num = dict()
    for num,value in enumerate(unique_values):
        str_to_num[value] = num

    #apply it to column
    train[col] = train[col].map(str_to_num)
    test[col]  = test[col].map(str_to_num)
    print(f'String values from {col} are transformed!')

print(' ')
print('Done!') 





##### Normalizing by replacing less frequents and NaNs values with the mode using CARD1

card_cols = ['card' + str(i) for i in range(2,7)]
normalize_columns('card1', card_cols, df_train=train, df_test=test, verbose=True) 

domain_cols = ['P_emaildomain', 'R_emaildomain']
normalize_columns('card1', domain_cols, df_train=train, df_test=test, verbose=True)





##### Replacing NaNs

# Cards columns
card_cols_c = ['card' + str(i) for i in range(1,7)]
fill_na(card_cols_c, num_rep=-999, obj_rep ='Unknown', df_train=train, df_test=test, verbose=True)

# Address columns
addrs = ['addr1', 'addr2']
fill_na(addrs, num_rep=-999, obj_rep ='Unknown', df_train=train, df_test=test, verbose=True)

# Domains columns
fill_na(domain_cols, num_rep=-999, obj_rep ='Unknown', df_train=train, df_test=test, verbose=True)





##### Creation of UIDs

# uid1

train['uid1'] = train['card1'].astype('str') + '_' + train['card2'].astype('str')     + '_'               + train['card3'].astype('str') + '_' + train['card4'].astype('str')     + '_'               + train['card5'].astype('str') + '_' + train['card6'].astype('str')     + '_' + train['addr1'].astype('str') + '_'               + train['addr2'].astype('str') + '_' + train['ProductCD'].astype('str') + '_' + train['D1achr'].astype('str')

test['uid1']  = test['card1'].astype('str')  + '_' + test['card2'].astype('str')     + '_'               + test['card3'].astype('str')  + '_' + test['card4'].astype('str')     + '_'               + test['card5'].astype('str')  + '_' + test['card6'].astype('str')     + '_' + test['addr1'].astype('str') + '_'               + test['addr2'].astype('str')  + '_' + test['ProductCD'].astype('str') + '_' + test['D1achr'].astype('str')

# uid2

train['uid2'] = train['card1'].astype('str')   + '_' + train['D2achr'].astype('str')  + '_'               + train['C13'].astype('str')     + '_' + train['D11achr'].astype('str') + '_'               + train['D10achr'].astype('str') + '_' + train['D15achr'].astype('str') + '_'               + train['D4achr'].astype('str')

test['uid2']  = test['card1'].astype('str')    + '_' + test['D2achr'].astype('str')   + '_'               + test['C13'].astype('str')      + '_' + test['D11achr'].astype('str')  + '_'               + test['D10achr'].astype('str')  + '_' + test['D15achr'].astype('str')  + '_'               + test['D4achr'].astype('str')

# uids

train['uids'] = train['card1'].astype('str')   + '_' + train['addr1'].astype('str')  + '_'               + train['D1achr'].astype('str')

test['uids']  = test['card1'].astype('str')    + '_' + test['addr1'].astype('str')   + '_'               + test['D1achr'].astype('str')

print('Unique uid1 (train):', len(train.uid1.unique()))
print('Unique uid1 (test):', len(test.uid1.unique()))

print('Unique uid2 (train):', len(train.uid2.unique()))
print('Unique uid2 (test):', len(test.uid2.unique()))

print('Unique uids (train):', len(train.uids.unique()))
print('Unique uids (test):', len(test.uids.unique()))





##### Selecting type of uid filling type
uid = 'uid1'





get_ipython().run_cell_magic('time', '', "##### Normalizing by replacing less frequents and NaNs values with the mode using UID1\n\nm_cols = ['M'+str(i) for i in range(1,10)]\nnormalize_columns(uid, m_cols, df_train=train, df_test=test, verbose=True) \nfill_na(m_cols, num_rep=-999, obj_rep ='Unknown', df_train=train, df_test=test, verbose=True)")





get_ipython().run_cell_magic('time', '', "##### Normalizing by replacing less frequents and NaNs values with the mode using UID1\n\ndist_cols = ['dist1', 'dist2']\nnormalize_columns(uid, dist_cols, df_train=train, df_test=test, verbose=True)\nfill_na(dist_cols, num_rep=-999, obj_rep ='Unknown', df_train=train, df_test=test, verbose=True)")





##### Memory reduction

train = mr(train, verbose=True)
test  = mr(test, verbose=True)





##### Saving both DataFrames into binary files to speed next uploadings.

print('Saving datasets...')
train.to_pickle('train.pkl')
print('Train has been saved... (1/2)')
test.to_pickle('test.pkl')
print('Test has been saved... (2/2)')
print('Done!')

