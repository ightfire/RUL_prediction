import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


train = pd.read_csv('', sep=" ", header=None)
test_df = pd.read_csv('', sep=" ", header=None)
train = np.array(train)
train = train[:, 1:-1]
test = np.array(test_df)
time = test[:, -2:].reshape(-1, 2)
test = np.delete(test, -2, axis=1)
p = test[:, :1].reshape(-1, 1)
print(test.shape)
test = test[:, 1:-1]
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler.transform(test)
test = np.hstack((p, test))
test = np.hstack((test, time))
test = pd.DataFrame(test)
test.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6',  's12', 's13', 's14', 's15', 's16', 's17',
                    's18', 's19', 'time', 's21']
print(test)
data = test[(test['time'] > 120)]#test['time'] <= 120)]    #Modify according to operating cycle
data = data.sort_values(by=['id', 's21'], ascending=[True, False])
print(data)
data.to_csv('', sep=" ", header=None, index=None)