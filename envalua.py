import math
import sklearn.metrics as skm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def evaluate_forecasts(actual, predicted):
    mse = skm.mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse

def evaluate_forecasts2(actual, predicted):
    err = predicted-actual
    score = 0
    for error in err:
        print(error)
        if error < 0:
            score = math.exp(-(error / 13)) - 1 + score
        if error >= 0:
            score = math.exp(error / 10) - 1 + score
    return score



if __name__ == '__main__':
    data = pd.read_csv('', header=None, sep=" ")
    data.columns = ['id',  'rul', 'predicted']
    data.loc[data.rul >= 130, 'rul'] = 130
    df = data.drop_duplicates(subset=('id'), keep='last')
    df = df.sort_values(by='rul')
    print(df)
    print(df.shape)
    data1 = np.array(df)
    predict = data1[:, -1].reshape(-1, 1)
    actual = data1[:, -2].reshape(-1, 1)
    scores = evaluate_forecasts(actual, predict)     # Different metrics can be called according to needs
    print(scores)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.plot(actual, color='blue', label='Actual RUL', marker='o', markersize=4)
    plt.plot(predict, color='red', label='Prediction', marker='o', markersize=4)
    my_x_ticks = np.arange(0, 250, 50)
    plt.yticks(my_x_ticks)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel('RUL', size=15)
    plt.xlabel('engine number', size=15)
