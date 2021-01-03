import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")





##### Functions
# 1st function: to graph time series based on TransactionDT vs the variable selected

def scatter(column):
    fr,no_fr = (train[train['isFraud'] == 1], train[train['isFraud'] == 0])  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3)) 
    ax1.title.set_text('Histogram ' + column + ' when isFraud == 0')
    ax1.set_ylim(train[column].min() - 1,train[column].max() + 1)
    ax1.scatter(x = no_fr['TransactionDT'], y = no_fr[column], color = 'blue', marker='o')   
    ax2.title.set_text('Histogram ' + column + ' when isFraud == 1')
    ax2.set_ylim(train[column].min() - 1,train[column].max() + 1)
    ax2.scatter(x = fr['TransactionDT'], y = fr[column], color = 'red', marker='o')
    plt.show()
    
# 2nd function: to show a ranking of pearson correlation with the variable selected

def corr(data,column):
    print('Correlation with ' + column)
    print(train[data].corrwith(train[column]).abs().sort_values(ascending = False)[1:])
    
# 3rd function: to reduce the groups based on Nans agroupation and pearson correlation

def reduce(groups):
    result = list()   
    for values in groups:
        maxval = 0
        val = values[0]  
        for value in values:
            unique_values = train[value].nunique()
            if unique_values > maxval:
                maxval = unique_values
                val = value 
        result.append(value)
    return result

# 4th function: to sort each column in ascending order based on its number

def order_finalcolumns(final_Xcolumns):
    return sorted(final_Xcolumns, key=lambda x: int("".join([i for i in x if i.isdigit()])))





##### Download of files.

print('Downloading datasets...')
print(' ')
train = pd.read_pickle('/kaggle/input/1-fraud-detection-memory-reduction/train_mred.pkl')
print('Train has been downloaded... (1/2)')
test = pd.read_pickle('/kaggle/input/1-fraud-detection-memory-reduction/test_mred.pkl')
print('Test has been downloaded... (2/2)')
print(' ')
print('All files are downloaded')





##### All the columns of train dataset.

print(list(train))


# # NaNs Exploration
# We will search all the columns to determine which columns are related by the number of NANs present. After grouping them, we decide to keep the columns of each group with major amount of unique values (its supposed to be the most explanatory variable)

# ## Transaction columns




# These columns are the first ones in transaction dataset.

columns= list(train.columns[:17])
columns





for col in columns:
    print(f'{col} NaNs: {train[col].isna().sum()} | {train[col].isna().sum()/train.shape[0]:.2%}')





# If we look closely to % NaNs data, most of them have low number of missing information. We are keeping all the columns where % NaNs < 0.7

final_transactioncolumns = list()
for col in columns:
    if train[col].isna().sum()/train.shape[0] < 0.7:
        final_transactioncolumns.append(col)
print('Final Transaction columns:',final_transactioncolumns)


# ## C columns




##### Group the C columns to determine which columns are related by the number of NANs present and analyze its groups independently.

columns = ['C' + str(i) for i in range(1,15)]
df_nan = train.isna()
dict_nans = dict()

for column in columns:
    number_nans = df_nan[column].sum()
    try:
        dict_nans[number_nans].append(column)
    except:
        dict_nans[number_nans] = [column]

group_number = 1
for key,values in dict_nans.items():
    print('Group {}'.format(group_number),'| Number of NANs =',key)
    print(values)
    print(' ')
    group_number += 1


# ### Group 1 (single group)




##### Time series graph based on TransactionDT
# There is no column that does not have NaNs values so we get all the columns in the same group

group_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['C1','C11','C2','C6','C8','C4','C10','C14','C12','C7','C13'], ['C3'], ['C5','C9']]
                 
result = reduce(reduce_groups)
print('Final C columns:',result)
final_ccolumns = result


# ## D columns




##### Group the D columns + Dachr columns to determine which columns are related by the number of NANs present and analyze its groups independently.

columns = ['D' + str(i) for i in range(1,16)]
columns.extend(['D1achr','D2achr','D4achr','D6achr','D10achr','D11achr','D12achr','D13achr','D14achr','D15achr'])
df_nan = train.isna()
dict_nans = dict()

for column in columns:
    number_nans = df_nan[column].sum()
    try:
        dict_nans[number_nans].append(column)
    except:
        dict_nans[number_nans] = [column]

group_number = 1
for key,values in dict_nans.items():
    print('Group {}'.format(group_number),'| Number of NANs =',key)
    print(values)
    print(' ')
    group_number += 1


# ### Group 1 (single group)




##### Time series graph based on TransactionDT.
# Despite having different number of NaNs, we are analyzing it as a single group. But due to NaNs low number in D1, we keep it as a final column.

group_list = ['D1achr', 'D2achr', 'D3', 'D4achr', 'D5', 'D6achr', 'D7', 'D8', 'D9', 'D10achr', 'D11achr', 'D12achr', 'D13achr', 'D14achr', 'D15achr']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7
# On the first group, D1achr vs D2achr --> we keep D1achr due to the low number of NaNs.

reduce_groups = [['D3','D7','D5'],['D4achr','D12achr','D6achr','D15achr','D10achr', 'D11achr'], ['D8'], ['D9'], ['D13achr'],['D14achr']]
                 
result = reduce(reduce_groups)
result.append('D1achr')
print('Final D columns:',result)
final_dcolumns = result


# ## M columns




##### Group the M columns to determine which columns are related by the number of NANs present and analyze its groups independently.

columns = ['M' + str(i) for i in range(1,10)]
df_nan = train.isna()
dict_nans = dict()

for column in columns:
    number_nans = df_nan[column].sum()
    try:
        dict_nans[number_nans].append(column)
    except:
        dict_nans[number_nans] = [column]

group_number = 1
for key,values in dict_nans.items():
    print('Group {}'.format(group_number),'| Number of NANs =',key)
    print(values)
    print(' ')
    group_number += 1


# ### Group 1 (single group)




# To analize M columns, we need to transform strings to numbers. Instead of using Label Encoder, we use a dictionary.

T_F_num = dict({'F': 0, 'T': 1, 'M0': 0, 'M1': 1, 'M2': 2})

for column in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    print(f'{column}:', train[column].unique())
    print('Transforming strings to numbers...')
    train[column] = train[column].replace(T_F_num)
    print(f'{column}:', train[column].unique())
    print('')





##### Time series graph based on TransactionDT.
# Despite having different number of NaNs, we are analyzing it as a single group.

group_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





#### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, We grouped together the columns with corr > 0.7 but in this case, no correlation is bigger than 0.7
# That's why, in this particular case we grouped together the columns with corr > 0.5

reduce_groups = ['M1'], ['M2','M3'], ['M4'], ['M5'], ['M6'], ['M7', 'M8'], ['M9']
                 
result = reduce(reduce_groups)
print('Final M columns:',result)
final_mcolumns = result


# ## V columns




##### Group the V columns to determine which columns are related by the number of NANs present and analyze its groups independently.

columns = ['V' + str(i) for i in range(1,340)]
df_nan = train.isna()
dict_nans = dict()

for column in columns:
    number_nans = df_nan[column].sum()
    try:
        dict_nans[number_nans].append(column)
    except:
        dict_nans[number_nans] = [column]

group_number = 1
for key,values in dict_nans.items():
    print('Group {}'.format(group_number),'| Number of NANs =',key)
    print(values)
    print(' ')
    group_number += 1
    
final_vcolumns = list()


# ### Group 1




##### Time series graph based on TransactionDT.
group_list = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = ['V1'], ['V2','V3'], ['V4','V5'], ['V6','V7'], ['V8','V9']

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group1 columns:',result)


# ### Group 2




##### Time series graph based on TransactionDT.

group_list = ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
              'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V12','V13'], ['V14'], ['V15','V16','V33','V34','V31','V32','V21','V22','V17','V18'], ['V19','V20'],['V23','V24'],['V25','V26'],['V27','V28'],['V29','V30']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group2 columns:',result)


# ### Group 3




##### Time series graph based on TransactionDT.

group_list = ['V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V35','V36'], ['V37','V38'], ['V39','V40','V42','V43','V50','V51','V52'], ['V41'], ['V44','V45'],['V46','V47'],['V48','V49']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group3 columns:',result)


# ### Group 4




##### Time series graph based on TransactionDT.

group_list = ['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 
              'V69', 'V70', 'V71', 'V72', 'V73', 'V74']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V53','V54'], ['V55','V56'], ['V57','V58','V71','V73','V72','V74','V63','V59','V64','V60'],['V61','V62'],['V65'],
                ['V66','V67'],['V68'], ['V69','V70']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group4 columns:',result)


# ### Group 5




##### Time series graph based on TransactionDT.

group_list = ['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V75','V76'],['V77','V78'], ['V79', 'V94', 'V93', 'V92', 'V84', 'V85', 'V80', 'V81'],['V82','V83'],['V86','V87'],['V88'],['V89'],['V90','V91']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group5 columns:',result)


# ### Group 6




##### Time series graph based on TransactionDT.

group_list = ['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 
              'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130',
              'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7
# We omit V107 since there is no info about corr with other columns and its unique values are 1.

reduce_groups = [['V95','V101'],['V96','V102','V97','V99','V100','V103'],['V98'],['V104','V106','V105'],['V108','V110','V114','V109','V111','V113','V112','V115','V116'],
                ['V117','V119','V118'],['V120','V122','V121'],['V123','V125','V124'],['V126','V128','V132'],['V127','V133','V134'],['V129','V131','V130'],
                ['V135','V137','V136']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group6 columns:',result)


# ### Group 7




##### Time series graph based on TransactionDT.

group_list = ['V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 
              'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V138'],['V139','V140'],['V141','V142'],['V143','V159','V150','V151','V165','V144','V145','V160','V152','V164','V166'],['V146','V147'],
                ['V148','V155','V149','V153','V154','V156','V157','V158'],['V161','V163','V162']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group7 columns:',result)


# ### Group 8




##### Time series graph based on TransactionDT.

group_list = ['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V186', 'V187', 'V190', 'V191', 'V192', 'V193', 
              'V196', 'V199', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = ['V167','V176','V199','V179','V190','V177','V186','V168','V172','V178','V196','V191','V204','V213','V207','V173'],['V181','V183','V182',
                'V187','V192','V203','V215','V178','V193','V212','V204'],['V202','V216','V204','V214']

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group8 columns:',result)


# ### Group 9




##### Time series graph based on TransactionDT.

group_list = ['V169', 'V170', 'V171', 'V174', 'V175', 'V180', 'V184', 'V185', 'V188', 'V189', 'V194', 'V195', 'V197', 'V198', 'V200', 'V201', 'V208', 'V209', 'V210']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V169'],['V170','V171','V200','V201'],['V174','V175'],['V180'],['V184','V185'],['V188','V189'],['V194','V197','V195','V198'],
                ['V208','V210','V209']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group9 columns:',result)


# ### Group 10




##### Time series graph based on TransactionDT.

group_list = ['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237','V240',
              'V241', 'V242', 'V243', 'V244', 'V246', 'V247', 'V248', 'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263',
              'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V217','V231','V233','V228','V257','V219','V232','V246'],['V218','V229','V224','V225','V253','V243','V254','V248','V264','V261','V249','V258',
                'V267','V274','V230','V236','V247','V262','V223','V252','V260'],['V226','V263','V276','V278'], ['V235','V237'],['V240','V241'],['V242','V244'],
                ['V265','V275','V277','V268','V273'],['V269','V266']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group10 columns:',result)


# ### Group 11




##### Time series graph based on TransactionDT.

group_list = ['V220', 'V221', 'V222', 'V227', 'V234', 'V238', 'V239', 'V245', 'V250', 'V251', 'V255', 'V256', 'V259', 'V270', 'V271', 'V272']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = ['V220'],['V221','V222','V259','V245','V227','V255','V256'],['V234'],['V238','V239'],['V250','V251'],['V270','V272','V271']

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group11 columns:',result)


# ### Group 12




##### Time series graph based on TransactionDT.

group_list = ['V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304',
              'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = [['V279','V293','V290','V280','V295','V294','V292','V291','V317','V307','V318'],['V284'],['V285','V287'],['V286'],['V297','V299','V298'],
                ['V302','V304','V303'],['V305'],['V306','V308','V316','V319'],['V309','V311','V312','V310'],['V320','V321']]

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group12 columns:',result)


# ### Group 13




##### Time series graph based on TransactionDT.

group_list = ['V281', 'V282', 'V283', 'V288', 'V289', 'V296', 'V300', 'V301', 'V313', 'V314', 'V315']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = ['V281','V282','V283'],['V288','V289'],['V296'],['V300','V301'],['V313','V315','V314']

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group13 columns:',result)


# ### Group 14




##### Time series graph based on TransactionDT.

group_list = ['V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']

for column in group_list:
    scatter(column)





##### Heatmap

plt.figure(figsize = (15,15))
sns.heatmap(train[group_list].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.show()





##### Ranking of pearson correlation.

for column in group_list:
    corr(group_list,column)
    print(' ')





##### Based on pearson correlation, we grouped together the columns with corr > 0.7

reduce_groups = ['V322','V324'],['V323','V326','V324','V327','V326'],['V325'],['V328','V330','V329'],['V331','V333','V332','V337'],['V334','V336','V335']

result = reduce(reduce_groups)
final_vcolumns.extend(result)
print('Final V_Group14 columns:',result)


# ### Final V columns




print('Number of V columns:', len(final_vcolumns))
print(final_vcolumns)


# # Conclusions
# Based on previous process, we suggest keeping as final columns the ones describes below:




##### 1st we sort them (ascending order) with a function

final_ccolumns = order_finalcolumns(final_ccolumns)
final_dcolumns = order_finalcolumns(final_dcolumns)
final_mcolumns = order_finalcolumns(final_mcolumns)
final_vcolumns = order_finalcolumns(final_vcolumns)





##### Final columns

print(f'Final Transaction columns ({len(final_transactioncolumns)}): {final_transactioncolumns}')
print(' ')
print(f'Final C columns ({len(final_ccolumns)}): {final_ccolumns}')
print(' ')
print(f'Final D columns ({len(final_dcolumns)}): {final_dcolumns}')
print(' ')
print(f'Final M columns ({len(final_mcolumns)}): {final_mcolumns}')
print(' ')
print(f'Final V columns ({len(final_vcolumns)}): {final_vcolumns}')
print(' ')

print('#' * 50)

final_columns = final_transactioncolumns + final_ccolumns + final_dcolumns + final_mcolumns + final_vcolumns
print(' ')
print('Final columns:', final_columns)
print(' ')
print('Lenght of final columns:', len(final_columns))







