import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression





LR = LinearRegression(fit_intercept=False)










if __name__ == '__main__':
    dataset = pd.read_csv('', header=None,sep=" ")
    data_test = pd.read_csv('', header=None,sep=" ")
    dataset = np.array(dataset)
    data_test = np.array(data_test)
    d, e = data_test[:, 0].reshape(-1, 1), data_test[:, -1:]
    data_n = np.hstack((d, e))
    print(dataset.shape)
    dataset1 = dataset[:, 1:-1]
    train_X = dataset1
    train_Y = dataset[:, -1]
    data_x = data_test[:, 1:-1]
    clf = LR
    clf.fit(train_X, train_Y)
    predicted = clf.predict(data_x)
    print(predicted.shape)
    predicted = predicted.reshape(-1,1)
    data = np.hstack((data_n, predicted))
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('', sep=" ", header=None, index=None)