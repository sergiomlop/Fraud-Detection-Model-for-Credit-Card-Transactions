# # Memory Reduction of our datasets
# 
# This notebook process both train and test dataframes downcasting the data types, allowing to work them safely in kaggle without running out of the assigned RAM memory. Also exports these files to binary, in order to speed their load in subsequent uses.
# 
# > For this task, we use the function created by Alexey Kupershtokh (https://www.kaggle.com/alexeykupershtokh/safe-memory-reduction)

 


import numpy as np 
import pandas as pd
from memory_reduction_script import reduce_mem_usage_sd as mr # we are using the memory reduction by our utility script


# ### This script will perform the following tasks:
# 
# * Creates the columns `DT`: the `TransactionDT` (Delta time) from seconds to days.
# * Creates the columns `Dxachr` referenced to the `DT` Delta days. These columns are days from a specific date or point in the past, so substracting the days passed since the begining of the dataset (`DT`), we can get that specific "date" delta.
# 
# We initially were not intended to work with test data, and we were partitioning or own test data from train for our final bootcamp project purposes.
# **Finally we decide to use full train for train and validation and test with test file, submitting the results to kaggle as we were participating in the competition.**

 


##### Download of files.

print('Downloading datasets...')
print(' ')
train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
print('Train has been downloaded... (1/2)')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
print('Test has been downloaded... (2/2)')
print('Done!')


 


##### Transforming the columns D en Dxachr (D anchored).
# D1, D2, D4, D6, D10, D11 - D15.
# D3, D5, D7 are differences between a transaction of a card and his previous one. Does not have sense to do the same thing.
# D8, D9 are probably hr. We keep them untouched.

train['DT'] = np.floor(train.TransactionDT/(60*60*24))
test['DT']  = np.floor(test.TransactionDT/(60*60*24))

columns_D = ['D' + str(i) for i in range(1,16) if i not in [3, 5, 7, 8, 9]]

for col in columns_D:
    train[col+'achr'] = train['DT'] - train[col]
    test[col+'achr']  = test['DT']  - test[col]


 


##### Reducing memory with safe downcast function.

train = mr(train, verbose=True)
test  = mr(test, verbose=True)


 


##### Saving both DataFrames into binary files to speed next uploadings.

print('Saving datasets...')
train.to_pickle('train_mred.pkl')
print('Train has been saved... (1/2)')
test.to_pickle('test_mred.pkl')
print('Test has been saved... (2/2)')
print('Done!')

