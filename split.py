import numpy as np
import pandas as pd

train_df = pd.read_csv('', sep=" ", header=None)
print(train_df.shape)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
print(train_df.shape)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                    's18', 's19', 's20', 's21']
train_df.drop(columns=['setting3','s1','s5','s10','s16','s18','s19'], axis=1,inplace=True)
train_df = train_df.sort_values(['id', 'cycle'])
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)
print(train_df)
temp = train_df.isnull().any()
print(type(temp))
print(temp)
train_df.to_csv('', sep=" ", header=None,index=None)


