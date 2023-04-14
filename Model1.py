from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt



#  Change the variable sw according to different window sizes_width
def sliding_window(train, sw_width, k, in_start=0,):
    data = train
    X, y, m = [], [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width

        if in_end <= len(data)-1:

            for i in range(in_start, in_end+1):
                x = data[in_start, 0]
                z = data[i, 0]
                if x != z :
                    in_start = i
                    in_end = in_start + sw_width
            train_seq = data[in_start:in_end, 1:-1]
            train_lab = data[in_end, -1]
            X.append(train_seq)
            m.append(data[in_end, 0])
            y.append(train_lab)
        in_start += k

    return np.array(X), np.array(y), np.array(m)





class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        # self.times.append(tf.timestamp() - self.timetaken)
        t=tf.timestamp() - self.timetaken
        self.times.append(t)
        print('cost time: ',t.numpy())
        self.epochs.append(epoch)
    def on_train_end(self,logs = {}):
        # plt.xlabel('Epoch')
        # plt.ylabel('Total time taken until an epoch in seconds')
        # plt.plot(self.epochs, self.times, 'ro')
        for i in range(len(self.epochs)):
          j = self.times[i].numpy()
          if i == 0:
            plt.text(i, j, str(round(j, 3)))
          else:
            j_prev = self.times[i-1].numpy()
            plt.text(i, j, str(round(j-j_prev, 3)))





def attention_model():
    inputs = Input(shape=(90, 18))
    cnn = Conv1D(filters=30, kernel_size=3, kernel_initializer='he_uniform', activation='relu', padding='same')(inputs)
    cnn = MaxPooling1D(pool_size=1)(cnn)
    gru_out = Bidirectional(GRU(units=30, activation='tanh'))(cnn)
    gru_out = Dense(30, activation='relu')(gru_out)
    output = Dense(1, activation='linear')(gru_out)
    model = Model(inputs=[inputs], outputs=output)
    return model

def stacking(x_train, y_train, x_test, k=5):
    kf = KFold(n_splits=k)
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_index], y_train[train_index]
        x_ts = x_train[test_index]
        print(y_tr.shape)
        model = attention_model()

        model.summary()
        model.compile(optimizer='adam', loss='mae')
        model.fit(x_tr, y_tr, epochs=50, batch_size=128, callbacks=[timecallback()], verbose=2)

        trainpre = model.predict(x_ts)
        testpre = model.predict(x_test)

        if (i == 0):
            trainprd = trainpre
            testprd = testpre
        else:
            trainprd = np.concatenate((trainprd, trainpre), axis=0)

            testprd += testpre
        i+=1

    testprd /=i

    return trainprd, testprd




if __name__ == '__main__':
    dataset = pd.read_csv('', header=None,sep=" ")
    data_test = pd.read_csv('', header=None,sep=" ")
    dataset = np.array(dataset)
    data_test = np.array(data_test)
    data_test = np.delete(data_test, -2, axis=1)
    print(dataset.shape)
    print(data_test.shape)
    data_n, y, m = sliding_window(data_test, sw_width=90, in_start=0, k=1)
    data_n2, y2, m2 = sliding_window(dataset, sw_width=90, in_start=0, k=1)
    print(m.shape)
    data_n2 = m2.reshape(-1, 1)
    data_n = m.reshape(-1, 1)
    y = y.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    print(y.shape)
    data_n = np.hstack((data_n, y))
    data_n2 = np.hstack((data_n2, y2))
    print(data_n.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset1 = scaler.fit_transform(dataset[:, 1:-1])
    print(dataset1.shape)
    dataset_a = dataset[:, :1]
    dataset_r = dataset[:, -1].reshape(-1, 1)
    dataset = np.hstack((dataset_a, dataset1))
    dataset = np.hstack((dataset, dataset_r))
    train_X, train_Y, m1 = sliding_window(dataset, sw_width=90, in_start=0, k=1)
    data_x, data_y, m2 = sliding_window(data_test, sw_width=90, in_start=0, k=1)
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    train_Y = scaler1.fit_transform(train_Y.reshape(-1, 1))
    train_pre, test_pre = stacking(train_X, train_Y, data_x)
    train_pre = train_pre.reshape(-1, 1)
    test_pre = test_pre.reshape(-1, 1)
    test_pre = scaler1.inverse_transform(test_pre)
    train_pre = scaler1.inverse_transform(train_pre)
    data_train = np.hstack((data_n2, train_pre))
    data = np.hstack((data_n, test_pre))
    df2 = pd.DataFrame(data_train)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('', sep=" ", header=None, index=None)
    df2.to_csv('', sep=" ", header=None, index=None)