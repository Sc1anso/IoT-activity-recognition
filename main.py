import json
import os.path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dropout, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D, Conv2D
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.lines as mlines

acc_anova_avg_lst = []
acc_anova_min_lst = []
n_feat_anova_avg_lst = []
n_feat_anova_min_lst = []
anova_val_tested_global = []
plot_labels_lst = []
anova_nof_avg_global = []
anova_acc_avg_global = []
y = list()
new_y_test = list()

train_iter_lst = [2]  # , 250, 500, 750, 1000, 5000, 10000, 100000
som_dim_lst = [10, 15, 20, 30]
# train_iter = 0

save_data = 'y'
w_path = 'weights'
plots_path = 'plots'
mod_path = 'som_models'
np_arr_path = 'np_arr'
min_som_dim = 10
act_som_dim = min_som_dim
old_som_dim = 0
max_som_dim = 50
step = 10
exec_n = 5
total_execs = 0
actual_exec = 0

if sys.argv[3] == 'uci':
    w_path = 'weights UCI'
    plots_path = 'plots UCI'
    mod_path = 'som_models UCI'
    np_arr_path = 'np_arr UCI'
if len(sys.argv) >= 6:
    if sys.argv[5] == 'uci':
        w_path = 'weights UCI'
        plots_path = 'plots UCI'
        mod_path = 'som_models UCI'
        np_arr_path = 'np_arr UCI'
if len(sys.argv) >= 7:
    if sys.argv[6] == 'n':
        save_data = 'n'
    elif sys.argv[6] == 'os':
        save_data = 'os'
    elif sys.argv[6] == 'anim':
        save_data = 'anim'
    elif sys.argv[6] == 'oc':
        save_data = 'oc'
elif sys.argv[4] == 'n':
    save_data = 'n'
if len(sys.argv) >= 8:
    min_som_dim = int(sys.argv[7])
if len(sys.argv) >= 9:
    max_som_dim = int(sys.argv[8])
if len(sys.argv) >= 10:
    step = int(sys.argv[9])
if len(sys.argv) >= 11:
    exec_n = int(sys.argv[10])

if save_data == 'anim' and sys.argv[3] == 'som':
    train_iter_lst.clear()
    for i_tmp in range(500, 15500, 500):
        train_iter_lst.append(i_tmp)

if not os.path.exists('./' + w_path):
    os.mkdir('./' + w_path)
if not os.path.exists('./' + plots_path):
    os.mkdir('./' + plots_path)
if not os.path.exists('./' + plots_path + '/anova_avg/'):
    os.mkdir('./' + plots_path + '/anova_avg/')
if not os.path.exists('./' + plots_path + '/anova_avg/som_bal_comp/'):
    os.mkdir('./' + plots_path + '/anova_avg/som_bal_comp')
if not os.path.exists('./' + plots_path + '/anova_avg/som_no-bal_comp/'):
    os.mkdir('./' + plots_path + '/anova_avg/som_no-bal_comp')
if not os.path.exists('./' + plots_path + '/anova_min/'):
    os.mkdir('./' + plots_path + '/anova_min/')
if not os.path.exists('./' + plots_path + '/anova_min/som_bal_comp/'):
    os.mkdir('./' + plots_path + '/anova_min/som_bal_comp')
if not os.path.exists('./' + plots_path + '/anova_min/som_no-bal_comp/'):
    os.mkdir('./' + plots_path + '/anova_min/som_no-bal_comp')
if not os.path.exists('./' + mod_path):
    os.mkdir('./' + mod_path)
if not os.path.exists('./' + mod_path + '/anova_avg/'):
    os.mkdir('./' + mod_path + '/anova_avg/')
if not os.path.exists('./' + mod_path + '/anova_min/'):
    os.mkdir('./' + mod_path + '/anova_min/')
if not os.path.exists('./' + np_arr_path):
    os.mkdir('./' + np_arr_path)
if not os.path.exists('./' + np_arr_path + '/anova_avg/'):
    os.mkdir('./' + np_arr_path + '/anova_avg/')
if not os.path.exists('./' + np_arr_path + '/anova_min/'):
    os.mkdir('./' + np_arr_path + '/anova_min/')

divider = 100000
range_lst = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
if 'UCI' in w_path:
    divider = 10000
    range_lst = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

total_execs = (((max_som_dim + step) - min_som_dim) / step) * exec_n * len(range_lst) * 2
if sys.argv[4] == 'avg' or sys.argv[4] == 'min':
    total_execs = (((max_som_dim + step) - min_som_dim) / step) * exec_n * len(range_lst)

def normalize(data):
    # Find the minimum and maximum value in the data
    min_val = np.min(data)
    max_val = np.max(data)

    # Calculate the range of the data
    data_range = max_val - min_val

    # Normalize the data
    normalized_data = (data - min_val) / data_range

    return normalized_data


# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, n_feat, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    if sys.argv[1] == "s":
        loaded = np.dstack(loaded)
    else:
        loaded = np.array(loaded[0][:, : n_feat])
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix, n_feat):
    if sys.argv[1] == "s":
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
        # body acceleration
        filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
        # body gyroscope
        filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    else:
        filepath = prefix + group + '/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['X_' + group + '.txt']
    # load input data
    X = load_group(filenames, n_feat, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix, n_feat):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + '/', n_feat)
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + '/', n_feat)
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = tf.keras.utils.to_categorical(trainy)
    print(min(testy))
    # input()
    testy = tf.keras.utils.to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    # input()
    if sys.argv[2] == "bal":
        return balance_data(trainX, trainy, testX, testy)
    else:
        return trainX, trainy, testX, testy


# balance data
def balance_data(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    bal_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    if sys.argv[1] == "s":
        X_train_bal = np.empty((0, X.shape[1], X.shape[2]))
        X_test_bal = np.empty((0, X.shape[1], X.shape[2]))
    else:
        X_train_bal = np.empty((0, X.shape[1]))
        X_test_bal = np.empty((0, X.shape[1]))
    y_train_bal = np.empty((0, y.shape[1]))
    y_test_bal = np.empty((0, y.shape[1]))
    for idx, item in enumerate(X):
        bal_val = 1500
        if len(sys.argv) >= 6:
            if sys.argv[5] == 'uci':
                bal_val = 1356
        if sys.argv[3] == 'uci':
            bal_val = 1356
        if bal_dict[int(np.argmax(y[idx]))] <= bal_val:
            X_train_bal = np.concatenate((X_train_bal, [item]))
            y_train_bal = np.concatenate((y_train_bal, [y[idx]]))
            bal_dict[int(np.argmax(y[idx]))] += 1
        else:
            X_test_bal = np.concatenate((X_test_bal, [item]))
            y_test_bal = np.concatenate((y_test_bal, [y[idx]]))

        print("\rBalancing data progress: " + str(idx + 1) + "/" + str(len(X)), end="")
    print()
    return X_train_bal, y_train_bal, X_test_bal, y_test_bal


# fit and evaluate LSTM model
def evaluate_lstm_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 100, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    if os.path.isfile('./' + w_path + '/lstm_' + str(sys.argv[2]) + '.h5'):
        model.load_weights('./' + w_path + '/lstm_' + str(sys.argv[2]) + '.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    checkpoint = ModelCheckpoint(filepath='./' + w_path + '/lstm_' + str(sys.argv[2]) + '.h5', monitor="val_loss",
                                 mode="min", save_best_only=True,
                                 verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[checkpoint, es],
              validation_split=.1)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    # print(model.summary())
    # input()
    return accuracy


# fit and evaluate cnn model
def evaluate_cnn_model(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 1, 100, 64
    X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((-1, X_train.shape[1], X_train.shape[2], 1))
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation='relu', input_shape=X_train[0].shape))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    if os.path.isfile('./' + w_path + '/cnn_' + str(sys.argv[2]) + '.h5'):
        model.load_weights('./' + w_path + '/cnn_' + str(sys.argv[2]) + '.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='./' + w_path + '/cnn_' + str(sys.argv[2]) + '.h5', monitor="val_loss",
                                 mode="min", save_best_only=True,
                                 verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[checkpoint, es],
              validation_split=.1)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return accuracy


# fit and evaluate CNN-LSTM model
def evaluate_cnnlstm_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 1, 100, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    # define model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    if os.path.isfile('./' + w_path + '/cnnlstm_' + str(sys.argv[2]) + '.h5'):
        model.load_weights('./' + w_path + '/cnnlstm_' + str(sys.argv[2]) + '.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='./' + w_path + '/cnnlstm_' + str(sys.argv[2]) + '.h5', monitor="val_loss",
                                 mode="min", save_best_only=True,
                                 verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[checkpoint, es],
              validation_split=.1)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    return accuracy


# fit and evaluate a model
def evaluate_convlstm_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 1, 100, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    # define model
    model = Sequential()
    model.add(
        ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.isfile('./' + w_path + '/convlstm_' + str(sys.argv[2]) + '.h5'):
        model.load_weights('./' + w_path + '/convlstm_' + str(sys.argv[2]) + '.h5')
    checkpoint = ModelCheckpoint(filepath='./' + w_path + '/convlstm_' + str(sys.argv[2]) + '.h5', monitor="val_loss",
                                 mode="min", save_best_only=True,
                                 verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[checkpoint, es],
              validation_split=.1)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    return accuracy


# take the second element for sort
def take_second(elem):
    return elem[1]


# take the first element for sort
def take_first(elem):
    return elem[0]


def classify(som, data, X_train, y_train, neurons, typ, a_val, train_iter):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """

    winmap = som.labels_map(X_train, y)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    if save_data == 'y':
        if not os.path.exists('./' + mod_path + '/anova_' + typ + '/' + str(a_val) + '/'):
            os.mkdir('./' + mod_path + '/anova_' + typ + '/' + str(a_val) + '/')
        final_map = {}
        global old_som_dim
        global act_som_dim
        if not old_som_dim == act_som_dim:
            for idx, val in enumerate(winmap):
                final_map.update({(val[0] * neurons) + val[1]: winmap[val].most_common()[0][0]})
                old_som_dim = act_som_dim
            # print("\nFINAL MAP\n")
            # print(final_map)
            final_map_lst = []
            pos_count = 0
            w_tot = pow(neurons, 2)
            for i in range(w_tot):
                if i not in final_map:
                    final_map.update({i: default_class})
            while len(final_map_lst) < len(final_map):
                for idx, val in enumerate(final_map):
                    if int(val) == pos_count:
                        final_map_lst.append(final_map[val])
                        pos_count += 1
                print("\rProgress: ", len(final_map_lst), '/', len(final_map), end='')
            final_map_lst = np.array(final_map_lst)
            if not os.path.exists('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/'):
                os.mkdir('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/')
            np.savetxt('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/map_lst_iter-' + str(train_iter) + '_' +
                       sys.argv[2] + '_' + str(neurons) + '.txt', final_map_lst, delimiter=' ')
            # input()
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def plot_som(som, X_train, y_train, path, a_val, n_feat, c_a):
    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map(scaling='mean').T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()

    # Plotting the response for each pattern in the iris dataset
    # different colors and markers for each label
    markers = ['o', 's', 'D', 'v', '1', 'P']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    activity = ['walking', 'w. upst', 'w. downst', 'sitting', 'standing', 'laying']
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[np.argmax(y_train[cnt])], markerfacecolor='None',
                 markeredgecolor=colors[np.argmax(y_train[cnt])], markersize=6, markeredgewidth=2,
                 label=activity[np.argmax(y_train[cnt])])
    mrk1 = mlines.Line2D([], [], markeredgecolor=colors[0], marker=markers[0], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    mrk2 = mlines.Line2D([], [], markeredgecolor=colors[1], marker=markers[1], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    mrk3 = mlines.Line2D([], [], markeredgecolor=colors[2], marker=markers[2], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    mrk4 = mlines.Line2D([], [], markeredgecolor=colors[3], marker=markers[3], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    mrk5 = mlines.Line2D([], [], markeredgecolor=colors[4], marker=markers[4], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    mrk6 = mlines.Line2D([], [], markeredgecolor=colors[5], marker=markers[5], markerfacecolor='None',
                         markeredgewidth=2, linestyle='None',
                         markersize=6)
    by_label = dict(zip(activity, [mrk1, mrk2, mrk3, mrk4, mrk5, mrk6]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    # plt.legend()
    # plt.show()
    if save_data == 'anim':
        plt.savefig(path + 'somAnimNoRand-' + str(c_a) + '.png')
    else:
        plt.savefig(path + str(a_val) + '_' + str(n_feat) + '.png')
    plt.close()


def get_anovaf(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    x_tmp = np.zeros(X.shape[0])
    x_medio = 0
    for idx, val in enumerate(X):
        for i, v in enumerate(val):
            x_tmp[i] += v
    for idx, val in enumerate(x_tmp):
        x_tmp[idx] /= X.shape[0]
    df = pd.DataFrame(X)
    df = df.T
    anova_prog_max = ((X.shape[1] * (X.shape[0] + 6)) * 2) + ((X.shape[1] * 6) * 2)
    anova_act_prog = 0
    elem_count = 0
    x_medio_lst = []
    x_medio_per_class = []
    for j in df.iterrows():
        x_medio_tmp = 0
        elem_count_tmp = 0
        class_medio = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            # dev_int += pow(j[1][i] - x_tmp[idx], 2)
            class_medio[np.argmax(y[i])] += 1
            x_medio += j[1][i]
            x_medio_tmp += j[1][i]
            elem_count += 1
            elem_count_tmp += 1
            anova_act_prog += 1
            # print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        for i in range(len(class_medio)):
            # print(elem_count_tmp)
            class_medio[i] /= elem_count_tmp
            anova_act_prog += 1
        print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        x_medio_per_class.append(class_medio)
        x_medio_lst.append(x_medio_tmp / elem_count_tmp)
    x_medio = x_medio / elem_count
    varianza_t_avgot = 0
    varianza_par_lst = []
    varianza_per_classe = []
    row_count = 0
    for j in df.iterrows():
        varianza_par = 0
        elem_count_tmp = 0
        class_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            class_dict[np.argmax(y[i])] += pow(j[1][i] - x_medio_per_class[row_count][np.argmax(y[i])], 2)
            varianza_t_avgot += pow(j[1][i] - x_medio, 2)
            varianza_par += pow(j[1][i] - x_medio, 2)
            elem_count_tmp += 1
            anova_act_prog += 1
            # print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        for i in range(len(class_dict)):
            class_dict[i] /= elem_count_tmp
            anova_act_prog += 1
        print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        varianza_per_classe.append(class_dict)
        varianza_par_lst.append(varianza_par / elem_count_tmp)
    varianza_t_avgot /= elem_count
    varianza_media_classi = []
    varianza_min_classi = []
    classe_varianza_min = []
    classi = 0
    for i in range(len(varianza_per_classe)):
        accumulatore = 0
        min_var_class = 1.0
        classe_min = 0
        for j in range(len(varianza_per_classe[i])):
            classi = len(varianza_per_classe[i])
            accumulatore += varianza_per_classe[i][j]
            if varianza_per_classe[i][j] < min_var_class:
                min_var_class = varianza_per_classe[i][j]
                classe_min = j
            anova_act_prog += 1
        print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        varianza_media_classi.append(accumulatore / classi)
        varianza_min_classi.append(min_var_class)
        classe_varianza_min.append(classe_min)
    for i in range(len(varianza_par_lst)):
        varianza_media_classi[i] /= varianza_par_lst[i]
        varianza_min_classi[i] /= varianza_par_lst[i]
        for j in range(len(varianza_per_classe[i])):
            varianza_per_classe[i][j] /= varianza_par_lst[i]
            anova_act_prog += 1
        print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")

    return varianza_media_classi, varianza_min_classi


def execute_minisom_anova(X_train, y_train, X_test, y_test, neurons, train_iter, c_anim, a_t_avg, a_t_min,
                          varianza_media_classi, varianza_min_classi):
    global old_som_dim
    global act_som_dim
    global exec_n
    global total_execs
    global actual_exec
    # features = load_file("./UCI HAR Dataset/features.txt")

    if sys.argv[4] == 'avg' or sys.argv[4] == 'avgmin':
        # CALCOLO RISULTATI CONSIDERANDO DIVERSI VALORI DI ANOVA AVG
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        features_names = {}
        n_neurons = 0
        for a_val in range_lst:
            less_t_anova_vals = []
            greater_t_anova_vals = []
            for idx, val in enumerate(varianza_media_classi):
                # print("F val feature " + str(idx) + ": " + str(val))
                if val > a_val / divider:
                    greater_t_anova_vals.append(idx)
                else:
                    less_t_anova_vals.append(idx)
            # X_ga = X_train[:, greater_t_anova_vals]
            X_la = X_train[:, less_t_anova_vals]
            n_neurons = m_neurons = neurons
            # m_neurons = 8
            som = MiniSom(n_neurons, m_neurons, X_la.shape[1], sigma=5, learning_rate=0.1,
                          neighborhood_function='gaussian', activation_distance='manhattan')

            if save_data == 'anim':
                som.pca_weights_init(X_la)
                som.train(X_la, train_iter, verbose=False)
            else:
                som.random_weights_init(X_la)
                som.train_random(X_la, train_iter, verbose=False)  # random training
            if not os.path.exists('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons)):
                os.mkdir('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons))
            if save_data == 'os' or save_data == 'y':
                plot_som(som, X_la, y_train, './' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons) + '/som_iter-' + str(train_iter) + '_plot_', a_val/divider, X_la.shape[1], c_anim)
            w = som.get_weights()
            w = w.reshape((-1, w.shape[2]))

            if not old_som_dim == act_som_dim:
                if save_data == 'y':
                    if not os.path.exists('./' + np_arr_path + '/anova_avg/' + str(a_val / divider) + '/'):
                        os.mkdir('./' + np_arr_path + '/anova_avg/' + str(a_val / divider) + '/')
                    np.savetxt('./' + np_arr_path + '/anova_avg/' + str(a_val / divider) + '/weights_lst_avg_iter-' + str(
                        train_iter) + '_' + sys.argv[2] + '_' + str(neurons) + '.txt', w, delimiter=' ')

                    if not os.path.exists('./' + mod_path + '/anova_avg/' + str(a_val / divider) + '/'):
                        os.mkdir('./' + mod_path + '/anova_avg/' + str(a_val / divider) + '/')
                old_som_dim = act_som_dim
            # print()
            class_repo = classification_report(new_y_test,
                                               classify(som, X_test[:, less_t_anova_vals], X_la, y_train, n_neurons,
                                                        'avg', a_val / divider, train_iter), output_dict=True)
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_la.shape[1])
            accuracies.append(class_repo['accuracy'])
            a_t_avg[a_val / divider].append(class_repo['accuracy'])
            actual_exec += 1
            percentage = round((actual_exec / total_execs) * 100, 2)
            print("\rProgress: ", percentage, "%", end="")
        if not os.path.exists('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons)):
            os.mkdir('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons))
        acc_anova_avg_lst.append(accuracies)
        anova_val_tested_global.append(anova_val_tested_str)
        n_feat_anova_avg_lst.append(n_feature_per_aval)
        plt.figure()
        plt.plot(anova_val_tested_str, accuracies, marker='o')
        plt.xlabel("Anova Threshold")
        plt.ylabel("mean of accuracies on 10 executions")
        plt.title("Accuracies comparison choosing the mean of the variances per class per f.")
        plt.close()
        plt.bar(anova_val_tested_str, n_feature_per_aval)
        plt.xlabel("anova val")
        plt.ylabel("n° features")
        plt.title("N° of features comparison choosing the mean of the variances per class per f.")
        plt.close()

    if sys.argv[4] == 'min' or sys.argv[4] == 'avgmin':
        # CALCOLO RISULTATI CONSIDERANDO DIVERSI VALORI DI ANOVA MIN
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        features_names = {}
        n_neurons = 0
        for a_val in range_lst:
            less_t_anova_vals = []
            greater_t_anova_vals = []
            for idx, val in enumerate(varianza_min_classi):
                # print("F val feature " + str(idx) + ": " + str(val))
                if val > a_val / divider:
                    greater_t_anova_vals.append(idx)
                else:
                    less_t_anova_vals.append(idx)
            # X_ga = X_train[:, greater_t_anova_vals]
            X_la = X_train[:, less_t_anova_vals]
            n_neurons = m_neurons = neurons
            # m_neurons = 8
            som = MiniSom(n_neurons, m_neurons, X_la.shape[1], sigma=5, learning_rate=0.1,
                          neighborhood_function='gaussian', activation_distance='manhattan')

            som.random_weights_init(X_la)
            som.train_random(X_la, train_iter, verbose=False)  # random training
            if not os.path.exists('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons)):
                os.mkdir('./' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_' + str(n_neurons))
            w = som.get_weights()
            w = w.reshape((-1, w.shape[2]))

            if not old_som_dim == act_som_dim:
                if save_data == 'y':
                    if not os.path.exists('./' + np_arr_path + '/anova_min/' + str(a_val / divider) + '/'):
                        os.mkdir('./' + np_arr_path + '/anova_min/' + str(a_val / divider) + '/')
                    np.savetxt('./' + np_arr_path + '/anova_min/' + str(a_val / divider) + '/weights_lst_min_iter-' + str(
                        train_iter) + '_' + sys.argv[2] + '_' + str(neurons) + '.txt', w,
                               delimiter=' ')

                    if not os.path.exists('./' + mod_path + '/anova_min/' + str(a_val / divider) + '/'):
                        os.mkdir('./' + mod_path + '/anova_min/' + str(a_val / divider) + '/')
                old_som_dim = act_som_dim
            class_repo = classification_report(new_y_test,
                                               classify(som, X_test[:, less_t_anova_vals], X_la, y_train, n_neurons,
                                                        'min', a_val / divider, train_iter), output_dict=True)
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_la.shape[1])
            accuracies.append(class_repo['accuracy'])
            a_t_min[a_val / divider].append(class_repo['accuracy'])
            actual_exec += 1
            percentage = round((actual_exec / total_execs) * 100, 2)
            print("\rProgress: ", percentage, "%", end="")
        if not os.path.exists('./' + plots_path + '/anova_min/som_' + sys.argv[2] + '_' + str(n_neurons) + '/'):
            os.mkdir('./' + plots_path + '/anova_min/som_' + sys.argv[2] + '_' + str(n_neurons) + '/')
        acc_anova_min_lst.append(accuracies)
        # anova_val_tested_global = anova_val_tested_str
        n_feat_anova_min_lst.append(n_feature_per_aval)
        plt.figure()
        plt.plot(anova_val_tested_str, accuracies, marker='o')
        plt.xlabel("Anova Threshold")
        plt.ylabel("mean of accuracies on 10 executions")
        plt.title("Accuracies comparison choosing the least of the variances per class per f.")
        # # plt.show()
        # if save_data == 'y':
        #     plt.savefig('./' + plots_path + '/anova_min/som_' + sys.argv[2] + '_' + str(n_neurons) + '/acc_comp_min_range(' + str(range_lst[0] / divider) + ',' + str(range_lst[len(range_lst)-1] / divider) + ')_iter-' + str(train_iter) + '.png')
        plt.close()
        plt.bar(anova_val_tested_str, n_feature_per_aval)
        plt.xlabel("Anova Threshold")
        plt.ylabel("n° features")
        plt.title("N° of features comparison choosing the least of the variances per class per f.")
        plt.close()


def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def execute_kmeans(X_train, y_train, X_test, y_test, m_iter, varianza_media_classi, varianza_min_classi,
                   acc_tot_avg=None):
    if acc_tot_avg is None:
        acc_tot_avg = {}
    pca = PCA(n_components=2)
    # X_pca[0] = norm_data(X_pca[0])
    # X_pca[1] = norm_data(X_pca[1])
    features = load_file("./UCI HAR Dataset/features.txt")
    # print("\nFEATURES\n")
    # print(features.shape)
    # input()
    kmeans = KMeans(init="random", n_clusters=6, max_iter=m_iter)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if sys.argv[4] == 'avg' or sys.argv[4] == 'avgmin':
        # CALCOLO RISULTATI CONSIDERANDO DIVERSI VALORI DI ANOVA AVG
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        features_names = {}
        for a_val in range_lst:
            less_t_anova_vals = []
            greater_t_anova_vals = []
            for idx, val in enumerate(varianza_media_classi):
                print("F val feature " + str(idx) + ": " + str(val))
                if val > a_val / divider:
                    greater_t_anova_vals.append(idx)
                else:
                    less_t_anova_vals.append(idx)
            print("count 0 <= i <= " + str(a_val / divider) + ": ", len(less_t_anova_vals))
            print(less_t_anova_vals)
            print("count > " + str(a_val / divider) + ": ", len(greater_t_anova_vals))
            print(greater_t_anova_vals)
            X_ga = X[:, greater_t_anova_vals]
            X_la = X[:, less_t_anova_vals]
            print("X_ga shape: ", X_ga.shape)
            print("X_la shape: ", X_la.shape)
            # input()
            X_pca = pca.fit_transform(X_la)
            X_pca = normalize(X_pca)
            # if save_data == 'y':
            #     np.save('./' + np_arr_path + '/X_la_avg_aval-' + str(a_val / divider) + '.npy', X_la)
            km_clusters = kmeans.fit_predict(X_pca)
            clusters_dict = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
            count = 0
            cluster_assoc_dict = {}
            for idx, val in enumerate(km_clusters):
                clusters_dict[val][np.argmax(y[idx])] += 1
            print()
            print(clusters_dict)
            print("\nCalcolo associazione clusters\n")
            for val in clusters_dict:
                print("cluster " + str(val) + ": label = " + str(max(clusters_dict[val], key=clusters_dict[val].get)) +
                      ", count = " + str(max(clusters_dict[val].values())) +
                      ", values = " + str(clusters_dict[val]))
                cluster_assoc_dict.update({val: max(clusters_dict[val], key=clusters_dict[val].get)})
            print("\nAssociazione cluster-label risultante\n")
            print(cluster_assoc_dict)
            # input()
            print("\nControllo cluster mancanti")
            found_lst = []
            dup_lst = []
            for val in cluster_assoc_dict:
                if cluster_assoc_dict[val] not in found_lst:
                    found_lst.append(cluster_assoc_dict[val])
                else:
                    dup_lst.append(cluster_assoc_dict[val])
            print("\nLabels presenti più di una volta\n")
            print(dup_lst)
            local_min_tmp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            while len(dup_lst) > 0:
                print("\nLabels presenti più di una volta\n")
                print(dup_lst)
                # input()
                cl_to_change = 0
                # local_min_tmp = 0
                for dup in dup_lst:
                    min_max_val = 0
                    for val in clusters_dict:
                        if max(clusters_dict[val], key=clusters_dict[val].get) == dup:
                            print("val: ", val, "cl_dict[val]: ", clusters_dict[val])
                            if min_max_val == 0:
                                min_max_val = max(clusters_dict[val].values())
                                cl_to_change = val
                            else:
                                if min_max_val > max(clusters_dict[val].values()):
                                    min_max_val = max(clusters_dict[val].values())
                                    cl_to_change = val
                    print("min max val: ", min_max_val)
                    print("cl to change: ", cl_to_change)
                    local_min = min(clusters_dict[cl_to_change].values())
                    # if local_min_tmp[dup] != 0:
                    #     min_max_val = local_min_tmp[dup]
                    new_label = 0
                    for val in clusters_dict[cl_to_change]:
                        if min_max_val > clusters_dict[cl_to_change][val] > local_min:
                            local_min = clusters_dict[cl_to_change][val]
                            local_min_tmp[dup] = local_min
                            new_label = val
                    print("local_min: ", local_min)
                    print("new label: ", new_label)
                    print("a bho: ", new_label in cluster_assoc_dict.values())
                    try:
                        cluster_assoc_dict[cl_to_change] = new_label
                    except Exception as e:
                        print(e)
                    if new_label in cluster_assoc_dict.values():  # and local_min == 0:
                        print("\nLA NUOVA LABEL NON è BUONA\n")
                        for i in range(0, 6, 1):
                            # print("i: ", i)
                            # print("new_label: ", new_label)
                            # print("i not in cluster assoc dict: ", i not in cluster_assoc_dict.values())
                            # input()
                            if i not in cluster_assoc_dict.values():
                                print(cluster_assoc_dict)
                                # cluster_assoc_dict.update({dup:new_label})
                                cluster_assoc_dict[cl_to_change] = i
                                clusters_dict[cl_to_change][i] = 10000
                                print(cluster_assoc_dict)
                                # input()
                print("\nNuova associazione clusters\n")
                print(cluster_assoc_dict)
                found_lst = []
                dup_lst = []
                # input()
                for val in cluster_assoc_dict:
                    # print("val: ", val)
                    # print("cluster assoc dict val: ", cluster_assoc_dict[val])
                    # input()
                    if cluster_assoc_dict[val] not in found_lst:
                        found_lst.append(cluster_assoc_dict[val])
                    else:
                        dup_lst.append(cluster_assoc_dict[val])
            if save_data == 'y':
                with open('./' + mod_path + '/cluster_assoc_dict_avg_aval-' + str(a_val / divider) + '.txt',
                          'w') as convert_file:
                    convert_file.write(json.dumps(cluster_assoc_dict))
            print("\nCalcolo accuracy\n")
            for idx, val in enumerate(km_clusters):
                if cluster_assoc_dict[val] == np.argmax(y[idx]):
                    count += 1
            print("acc: " + str(count / X_pca.shape[0]))
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_la.shape[1])
            anova_nof_avg_global.append(X_la.shape[1])
            accuracies.append(count / X_pca.shape[0])
            anova_acc_avg_global.append(count / X_pca.shape[0])
            features_names.update({a_val / divider: features[less_t_anova_vals, 1]})
            if sys.argv[3] == 'both':
                acc_tot_avg[a_val / divider].append(count / X_pca.shape[0])
        for idx, val in enumerate(anova_val_tested):
            print("val tested: ", val)
            print("features considered: ", n_feature_per_aval[idx])
            print("accuracy: ", accuracies[idx])
            # print("feature names: ", features_names[val])
        plt.figure()
        plt.plot(anova_val_tested_str, accuracies)
        plt.xlabel("anova val")
        plt.ylabel("accuracy")
        plt.title("Accuracies comparison choosing the mean of the variances per class per f.")
        # # plt.show()
        if save_data == 'y':
            plt.savefig('./' + plots_path + '/anova_avg/acc_comp_avg.png')
        plt.close()
        plt.bar(anova_val_tested_str, n_feature_per_aval)
        plt.xlabel("anova val")
        plt.ylabel("n° features")
        plt.title("N° of features comparison choosing the mean of the variances per class per f.")
        plt.close()

    if sys.argv[4] == 'min' or sys.argv[4] == 'avgmin':
        # CALCOLO RISULTATI CONSIDERANDO DIVERSI VALORI DI ANOVA MIN
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        features_names = {}
        for a_val in range_lst:
            less_t_anova_vals = []
            greater_t_anova_vals = []
            for idx, val in enumerate(varianza_min_classi):
                print("F val feature " + str(idx) + ": " + str(val))
                if val > a_val / divider:
                    greater_t_anova_vals.append(idx)
                else:
                    less_t_anova_vals.append(idx)
            X_ga = X[:, greater_t_anova_vals]
            X_la = X[:, less_t_anova_vals]
            X_pca = pca.fit_transform(X_la)
            X_pca = normalize(X_pca)
            km_clusters = kmeans.fit_predict(X_pca)
            clusters_dict = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                             5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
            count = 0
            cluster_assoc_dict = {}
            for idx, val in enumerate(km_clusters):
                clusters_dict[val][np.argmax(y[idx])] += 1
            for val in clusters_dict:
                cluster_assoc_dict.update({val: max(clusters_dict[val], key=clusters_dict[val].get)})
            found_lst = []
            dup_lst = []
            for val in cluster_assoc_dict:
                if cluster_assoc_dict[val] not in found_lst:
                    found_lst.append(cluster_assoc_dict[val])
                else:
                    dup_lst.append(cluster_assoc_dict[val])
            local_min_tmp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            while len(dup_lst) > 0:
                cl_to_change = 9
                for dup in dup_lst:
                    min_max_val = 0
                    for val in clusters_dict:
                        if max(clusters_dict[val], key=clusters_dict[val].get) == dup:
                            if min_max_val == 0:
                                min_max_val = max(clusters_dict[val].values())
                                cl_to_change = val
                            else:
                                if min_max_val > max(clusters_dict[val].values()):
                                    min_max_val = max(clusters_dict[val].values())
                                    cl_to_change = val
                    local_min = min(clusters_dict[cl_to_change].values())
                    new_label = 0
                    for val in clusters_dict[cl_to_change]:
                        if min_max_val > clusters_dict[cl_to_change][val] > local_min:
                            local_min = clusters_dict[cl_to_change][val]
                            local_min_tmp[dup] = local_min
                            new_label = val
                    try:
                        cluster_assoc_dict[cl_to_change] = new_label
                    except Exception as e:
                        print(e)
                    if new_label in cluster_assoc_dict.values():  # and local_min == 0:
                        # print("\nLA NUOVA LABEL NON è BUONA\n")
                        for i in range(0, 6, 1):
                            if i not in cluster_assoc_dict.values() and cl_to_change != 9:
                                cluster_assoc_dict[cl_to_change] = i
                                clusters_dict[cl_to_change][i] = 10000
                found_lst = []
                dup_lst = []
                # input()
                for val in cluster_assoc_dict:
                    if cluster_assoc_dict[val] not in found_lst:
                        found_lst.append(cluster_assoc_dict[val])
                    else:
                        dup_lst.append(cluster_assoc_dict[val])
            if save_data == 'y':
                with open('./' + mod_path + '/cluster_assoc_dict_min_aval-' + str(a_val / divider) + '.txt',
                          'w') as convert_file:
                    convert_file.write(json.dumps(cluster_assoc_dict))
            print("\nCalcolo accuracy\n")
            for idx, val in enumerate(km_clusters):
                if cluster_assoc_dict[val] == np.argmax(y[idx]):
                    count += 1
            print("acc: " + str(count / X_pca.shape[0]))
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_la.shape[1])
            accuracies.append(count / X_pca.shape[0])
            features_names.update({a_val / divider: features[less_t_anova_vals, 1]})
            # PLOT KMEANS RESULTS
            plot_km_fit_result(kmeans, X_pca, km_clusters, y, a_val / divider, "min", m_iter, cluster_assoc_dict)
        for idx, val in enumerate(anova_val_tested):
            print("val tested: ", val)
            print("features considered: ", n_feature_per_aval[idx])
            print("accuracy: ", accuracies[idx])
            # print("feature names: ", features_names[val])
        plt.figure()
        plt.plot(anova_val_tested_str, accuracies)
        plt.xlabel("anova val")
        plt.ylabel("accuracy")
        plt.title("Accuracies comparison choosing the least of the variances per class per f.")
        # plt.show()
        if save_data == 'y':
            plt.savefig('./' + plots_path + '/anova_min/acc_comp_min.png')
        plt.close()
        plt.bar(anova_val_tested_str, n_feature_per_aval)
        plt.xlabel("anova val")
        plt.ylabel("n° features")
        plt.title("N° of features comparison choosing the least of the variances per class per f.")
        # plt.show()
        if save_data == 'y':
            plt.savefig('./' + plots_path + '/anova_min/nof_comp_min.png')
        plt.close()

        plt.figure()
        plt.plot(anova_val_tested_str, anova_acc_avg_global, label='anova avg', marker='o')
        plt.plot(anova_val_tested_str, accuracies, label='anova min', marker='o')
        plt.xlabel("anova val")
        plt.ylabel("accuracy")
        plt.title("Accuracies comparison choosing between anova avg and min")
        plt.legend()
        # plt.show()
        if save_data == 'y':
            plt.savefig('./' + plots_path + '/acc_comp.png')
        plt.close()

        fig, ax = plt.subplots()
        bottom = np.zeros(len(anova_val_tested_str))
        new_min = []
        for idx, val in enumerate(anova_nof_avg_global):
            new_min.append(n_feature_per_aval[idx] - val)
        p = ax.bar(anova_val_tested_str, anova_nof_avg_global, label='AVG', bottom=bottom)
        ax.bar_label(p, padding=-15)
        bottom += np.array(anova_nof_avg_global)
        p = ax.bar(anova_val_tested_str, new_min, label='MIN', bottom=bottom)
        ax.bar_label(p, padding=3)
        # ax.set_title("N° of features comparison choosing between anova avg or min")
        ax.set_ylabel('# of features')
        ax.set_xlabel('Anova Threshold')
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 325)
        # plt.show()
        if save_data == 'y':
            plt.savefig('./' + plots_path + '/nof_comp.png')
        plt.close()


def plot_km_fit_result(kmeans, X, clusters, y_train, a_val, mode, frame_num, cl_a_d):
    """# Plotting KMeans results"""
    plt.figure(figsize=(16, 10))

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.001  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    colormap = {0: "b", 1: "g", 2: "r", 3: "c", 4: "m", 5: "k"}
    labelmap = {"b": "walking", "g": "upstairs", "r": "downstairs", "c": "sitting", "m": "standing", "k": "laying"}
    label_mapped = {"b": False, "g": False, "r": False, "c": False, "m": False, "k": False}
    color_list = []
    for idx, val in enumerate(X):
        color_list.append(colormap[np.argmax(y_train[idx])])
    color_list = np.array(color_list)
    for idx, val in enumerate(X):
        if label_mapped[color_list[idx]]:
            plt.plot(val[0], val[1], color=color_list[idx], marker=".", markersize=3)
        else:
            plt.plot(val[0], val[1], color=color_list[idx], marker=".", markersize=3, label=labelmap[color_list[idx]])
            label_mapped[color_list[idx]] = True
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    colormap_cen = {0: "w", 1: "w", 2: "w", 3: "w", 4: "w", 5: "w"}
    for idx, val in enumerate(centroids):
        plt.scatter(
            val[0],
            val[1],
            marker="x",
            s=180,
            linewidths=3,
            color=colormap_cen[idx],
            zorder=10,
        )
    plt.title(
        "K-means clustering\n"
        "Centroids are marked with white cross"
    )
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    # plt.show()
    if save_data == 'y':
        plt.savefig(
            './' + plots_path + '/anova_' + str(mode) + '/kmeans_anova_' + str(a_val) + '_' + str(mode) + '.png')
    if save_data == 'anim':
        plt.savefig('./' + plots_path + '/anova_' + str(mode) + '/anim/km-' + str(int(frame_num)) + '.png')
    plt.close()


def plot_som_comp(train_iter, accs_avg_mean, accs_min_mean, accs_avg_max, accs_min_max, accs_avg_min, accs_min_min,
                  acc_mean_km=None, acc_min_km=None, acc_max_km=None):
    if acc_min_km is None:
        acc_min_km = {}
    if acc_max_km is None:
        acc_max_km = {}
    if acc_mean_km is None:
        acc_mean_km = {}
    name = 'som'
    if sys.argv[3] == 'both':
        name = 'som-km'
    min_neurons = None
    if sys.argv[4] == 'avg' or sys.argv[4] == 'avgmin':
        plt.figure()
        # print(anova_val_tested_global)
        key_lst_km = []
        for k in accs_avg_mean.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_avg_mean[k].keys():
                keys_lst.append(str(val))
            for val in accs_avg_mean[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
            key_lst_km = keys_lst
            # plt.xticks(np.array(anova_val_tested_global))
        if sys.argv[3] == 'both':
            plt.plot(key_lst_km, acc_mean_km.values(), label='K-Means', marker='o')
        # plt.xticks(anova_val_tested_global[0])
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the mean of the variances per class per f."
        # plt.title(string)
        plt.legend()
        # plt.show()
        # step_val = 0
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y' or save_data == 'oc':
            plt.savefig(
                './' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_avg_mean_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(exec_n) + '.png')
        plt.close()
        plt.figure()
        # print(anova_val_tested_global)
        for k in accs_avg_max.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_avg_max[k].keys():
                keys_lst.append(str(val))
            for val in accs_avg_max[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
        # plt.xticks(anova_val_tested_global[0])
        if sys.argv[3] == 'both':
            plt.plot(key_lst_km, acc_max_km.values(), label='K-Means', marker='o')
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the mean of the variances per class per f."
        # plt.title(string)
        plt.legend()
        # plt.show()
        # step_val = 0
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y' or save_data == 'oc':
            plt.savefig(
                './' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_avg_max_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(
                    exec_n) + '.png')
        plt.close()
        plt.figure()
        # print(anova_val_tested_global)
        for k in accs_avg_min.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_avg_min[k].keys():
                keys_lst.append(str(val))
            for val in accs_avg_min[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
        # plt.xticks(anova_val_tested_global[0])
        if sys.argv[3] == 'both':
            plt.plot(key_lst_km, acc_min_km.values(), label='K-Means', marker='o')
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the mean of the variances per class per f."
        # plt.title(string)
        plt.legend()
        # plt.show()
        # step_val = 0
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y' or save_data == 'oc':
            plt.savefig(
                './' + plots_path + '/anova_avg/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_avg_min_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(
                    exec_n) + '.png')
        plt.close()
    if sys.argv[4] == 'min' or sys.argv[4] == 'avgmin':
        plt.figure()
        for k in accs_min_mean.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_min_mean[k].keys():
                keys_lst.append(str(val))
            for val in accs_min_mean[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
        # plt.xticks(anova_val_tested_global[0])
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the least of the variances per class per f."
        # plt.title(string)
        plt.legend()
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        # plt.show()
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y':
            plt.savefig(
                './' + plots_path + '/anova_min/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_min_mean_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(exec_n) + '.png')
        plt.close()
        plt.figure()
        # print(anova_val_tested_global)
        for k in accs_min_max.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_min_max[k].keys():
                keys_lst.append(str(val))
            for val in accs_min_max[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
        # plt.xticks(anova_val_tested_global[0])
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the least of the variances per class per f."
        # plt.title(string)
        plt.legend()
        # plt.show()
        # step_val = 0
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y':
            plt.savefig(
                './' + plots_path + '/anova_min/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_min_max_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(
                    exec_n) + '.png')
        plt.close()
        plt.figure()
        # print(anova_val_tested_global)
        for k in accs_min_min.keys():
            keys_lst = []
            vals_lst = []
            for val in accs_min_min[k].keys():
                keys_lst.append(str(val))
            for val in accs_min_min[k].values():
                vals_lst.append(val)
            plt.plot(keys_lst, vals_lst, label=str(k) + 'x' + str(k), marker='o')
        # plt.xticks(anova_val_tested_global[0])
        plt.xlabel("Anova Threshold")
        plt.ylabel("Accuracy")
        string = "Accuracies comparison choosing the least of the variances per class per f."
        # plt.title(string)
        plt.legend()
        # plt.show()
        # step_val = 0
        min_neurons = plot_labels_lst[0].split('x')[0]
        max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split('x')[0]
        step_neurons = 0
        if len(plot_labels_lst) > 1:
            step_val = plot_labels_lst[1].split('x')[0]
            step_neurons = int(step_val) - int(min_neurons)
        if save_data == 'y':
            plt.savefig(
                './' + plots_path + '/anova_min/som_' + sys.argv[2] + '_comp' + '/' + name + '_comp_min_min_iter-' + str(
                    train_iter) + '_range(' + str(
                    range_lst[0] / divider) + ',' + str(range_lst[len(range_lst) - 1] / divider) + ')_f-' + str(
                    min_neurons) + 't-' + str(max_neurons) + '_s-' + str(step_neurons) + '_execs-' + str(
                    exec_n) + '.png')
        plt.close()


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (float(m), float(s)))


# run an experiment
def run_experiment(repeats):
    global act_som_dim
    if sys.argv[1] == "s":
        dataset = 'Dataset tesi'
        if sys.argv[3] == 'uci':
            dataset = 'UCI HAR Dataset'
        #     w_path = 'weights UCI'
        trainX, trainy, testX, testy = load_dataset('./' + dataset, 265)
        # repeat experiment
        scores_lstm = list()
        scores_cnn = list()
        scores_cnnlstm = list()
        scores_convlstm = list()
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # input()
        for r in range(repeats):
            plots_val = []
            plots_lbl = ['LSTM', 'CNN', 'CNN-LSTM', 'CONVLSTM']
            print("LSTM")
            score = evaluate_lstm_model(trainX, trainy, testX, testy)
            score = score * 100.0
            plots_val.append(score)
            print('>#%d: %.3f' % (r + 1, score))
            scores_lstm.append(score)
            print("CNN")
            score = evaluate_cnn_model(trainX, trainy, testX, testy)
            score = score * 100.0
            plots_val.append(score)
            print('>#%d: %.3f' % (r + 1, score))
            scores_cnn.append(score)
            print("CNN-LSTM")
            score = evaluate_cnnlstm_model(trainX, trainy, testX, testy)
            score = score * 100.0
            plots_val.append(score)
            print('>#%d: %.3f' % (r + 1, score))
            scores_cnnlstm.append(score)
            print("CONV-LSTM")
            score = evaluate_convlstm_model(trainX, trainy, testX, testy)
            score = score * 100.0
            plots_val.append(score)
            print('>#%d: %.3f' % (r + 1, score))
            scores_convlstm.append(score)
            fig, ax = plt.subplots()
            bottom = np.zeros(len(plots_lbl))
            p = ax.bar(plots_lbl, plots_val, bottom=bottom)
            ax.bar_label(p, padding=3)
            ax.set_title('Confronto accuracies ottenute tra le tecniche supervised')
            ax.set_ylabel('Accuracies ottenute')
            ax.set_xlabel('Modelli testati')
            ax.set_ylim(0, 110)
            if save_data == 'y':
                plt.savefig('./' + plots_path + '/acc_comp_supervised_' + str(sys.argv[2]) + '.png')
            plt.close()
        # summarize results
        print("LSTM Results")
        summarize_results(scores_lstm)
        print("CNN Results")
        summarize_results(scores_cnn)
        print("CNN-LSTM Results")
        summarize_results(scores_cnnlstm)
        print("CONV-LSTM Results")
        summarize_results(scores_convlstm)
    else:
        global range_lst
        dataset = 'Dataset tesi'
        if sys.argv[5] == 'uci':
            dataset = 'UCI HAR Dataset'
        feat = [265]
        if sys.argv[4] == 'no':
            feat = [265, 561]
            range_lst = [10000]
        trainX, trainy, testX, testy = load_dataset('./' + dataset, feat[0])
        if sys.argv[3] == 'som':
            count_anim = 0
            for idx, item in enumerate(trainy):
                y.append(np.argmax(trainy[idx]))
            for idx, item in enumerate(testy):
                new_y_test.append(np.argmax(testy[idx]))
            for t_iter in train_iter_lst:
                acc_anova_avg_lst.clear()
                acc_anova_min_lst.clear()
                n_feat_anova_avg_lst.clear()
                n_feat_anova_min_lst.clear()
                plot_labels_lst.clear()
                anova_nof_avg_global.clear()
                anova_acc_avg_global.clear()
                v_avg_c, v_min_c = get_anovaf(trainX, trainy, testX, testy)
                print()
                accs_min_mean = {10: {}}
                accs_min_max = {10: {}}
                accs_min_min = {10: {}}
                accs_avg_mean = {10: {}}
                accs_avg_max = {10: {}}
                accs_avg_min = {10: {}}
                if sys.argv[4] == 'avg':
                    for i in range(min_som_dim, max_som_dim + step, step):
                        if 'UCI' in w_path:
                            accs_min_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                      0.8: [], 0.9: [], 1.0: []}})
                            accs_min_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                     0.8: [], 0.9: [], 1.0: []}})
                            accs_min_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                     0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                      0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                     0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [],
                                                     0.8: [], 0.9: [], 1.0: []}})

                        if 'UCI' not in w_path:
                            accs_min_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                    for i in range(min_som_dim, max_som_dim + step, step):
                        act_som_dim = i
                        accs_tot_min = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        accs_tot_avg = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        if 'UCI' not in w_path:
                            accs_tot_avg = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                            accs_tot_min = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                        plot_labels_lst.append(str(i) + 'x' + str(i))
                        for j in range(1, exec_n + 1, 1):
                            anova_val_tested_global.clear()
                            execute_minisom_anova(X_train=trainX, y_train=trainy, X_test=testX, y_test=testy,
                                                  neurons=i, train_iter=t_iter, c_anim=count_anim, a_t_avg=accs_tot_avg,
                                                  a_t_min=accs_tot_min, varianza_media_classi=v_avg_c,
                                                  varianza_min_classi=v_min_c)
                        accs_tot_min_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                            0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                            0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                            0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                            0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                             0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                             0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        if 'UCI' not in w_path:
                            accs_tot_min_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                        for k in accs_tot_avg.keys():
                            accs_tot_avg_mean.update({k: np.mean(accs_tot_avg[k])})
                            accs_tot_avg_max.update({k: np.max(accs_tot_avg[k])})
                            accs_tot_avg_min.update({k: np.min(accs_tot_avg[k])})
                        accs_avg_mean.update({i: accs_tot_avg_mean})
                        accs_avg_max.update({i: accs_tot_avg_max})
                        accs_avg_min.update({i: accs_tot_avg_min})
                else:
                    for i in range(min_som_dim, max_som_dim + step, step):
                        if 'UCI' in w_path:
                            accs_min_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_min_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_min_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})

                        if 'UCI' not in w_path:
                            accs_min_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                    for i in range(min_som_dim, max_som_dim + step, step):
                        act_som_dim = i
                        accs_tot_min = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        accs_tot_avg = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        if 'UCI' not in w_path:
                            accs_tot_avg = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                            accs_tot_min = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                        plot_labels_lst.append(str(i) + 'x' + str(i))
                        for j in range(1, exec_n + 1, 1):
                            anova_val_tested_global.clear()
                            execute_minisom_anova(X_train=trainX, y_train=trainy, X_test=testX, y_test=testy,
                                                  neurons=i, train_iter=t_iter, c_anim=count_anim, a_t_avg=accs_tot_avg,
                                                  a_t_min=accs_tot_min, varianza_media_classi=v_avg_c,
                                                  varianza_min_classi=v_min_c)
                        accs_tot_min_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        if 'UCI' not in w_path:
                            accs_tot_min_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                        for k in accs_tot_avg.keys():
                            accs_tot_avg_mean.update({k: np.mean(accs_tot_avg[k])})
                            accs_tot_min_mean.update({k: np.mean(accs_tot_min[k])})
                            accs_tot_avg_max.update({k: np.max(accs_tot_avg[k])})
                            accs_tot_min_max.update({k: np.max(accs_tot_min[k])})
                            accs_tot_avg_min.update({k: np.min(accs_tot_avg[k])})
                            accs_tot_min_min.update({k: np.min(accs_tot_min[k])})
                        accs_avg_mean.update({i: accs_tot_avg_mean})
                        accs_min_mean.update({i: accs_tot_min_mean})
                        accs_avg_max.update({i: accs_tot_avg_max})
                        accs_min_max.update({i: accs_tot_min_max})
                        accs_avg_min.update({i: accs_tot_avg_min})
                        accs_min_min.update({i: accs_tot_min_min})
                plot_som_comp(t_iter, accs_avg_mean, accs_min_mean, accs_avg_max, accs_min_max, accs_avg_min, accs_min_min)
                count_anim += 1
        elif sys.argv[3] == 'both':
            if sys.argv[4] == 'avg' or sys.argv[4] == 'min' or sys.argv[4] == 'avgmin':
                count_anim = 0
                for idx, item in enumerate(trainy):
                    y.append(np.argmax(trainy[idx]))
                for idx, item in enumerate(testy):
                    new_y_test.append(np.argmax(testy[idx]))
                for t_iter in train_iter_lst:
                    acc_anova_avg_lst.clear()
                    acc_anova_min_lst.clear()
                    n_feat_anova_avg_lst.clear()
                    n_feat_anova_min_lst.clear()
                    plot_labels_lst.clear()
                    anova_nof_avg_global.clear()
                    anova_acc_avg_global.clear()
                    v_avg_c, v_min_c = get_anovaf(trainX, trainy, testX, testy)
                    print()
                    accs_min_mean = {min_som_dim: {}}
                    accs_min_max = {min_som_dim: {}}
                    accs_min_min = {min_som_dim: {}}
                    accs_avg_mean = {min_som_dim: {}}
                    accs_avg_max = {min_som_dim: {}}
                    accs_avg_min = {min_som_dim: {}}
                    for i in som_dim_lst:
                        if 'UCI' in w_path:
                            accs_min_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_min_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_min_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_max.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})
                            accs_avg_min.update({i: {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}})

                        if 'UCI' not in w_path:
                            accs_min_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_min_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_mean.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                      0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_max.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                            accs_avg_min.update({i: {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                     0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}})
                    for i in som_dim_lst:
                        act_som_dim = i
                        accs_tot_min = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        accs_tot_avg = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                        1.0: []}
                        if 'UCI' not in w_path:
                            accs_tot_avg = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                            accs_tot_min = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [], 0.01: [],
                                            0.05: [], 0.1: [], 0.5: [], 1.0: []}
                        plot_labels_lst.append(str(i) + 'x' + str(i))
                        for j in range(1, exec_n + 1, 1):
                            anova_val_tested_global.clear()
                            execute_minisom_anova(X_train=trainX, y_train=trainy, X_test=testX, y_test=testy,
                                                  neurons=i, train_iter=t_iter, c_anim=count_anim, a_t_avg=accs_tot_avg,
                                                  a_t_min=accs_tot_min, varianza_media_classi=v_avg_c,
                                                  varianza_min_classi=v_min_c)
                        accs_tot_min_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_min = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_max = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_min_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        accs_tot_avg_mean = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                         0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                        if 'UCI' not in w_path:
                            accs_tot_min_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_min = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_max = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_min_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                            accs_tot_avg_mean = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                 0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                        for k in accs_tot_avg.keys():
                            accs_tot_avg_mean.update({k: np.mean(accs_tot_avg[k])})
                            accs_tot_avg_max.update({k: np.max(accs_tot_avg[k])})
                            accs_tot_avg_min.update({k: np.min(accs_tot_avg[k])})
                        accs_avg_mean.update({i: accs_tot_avg_mean})
                        accs_avg_max.update({i: accs_tot_avg_max})
                        accs_avg_min.update({i: accs_tot_avg_min})
                    accs_tot_avg_km = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [],
                                    1.0: []}
                    if 'UCI' not in w_path:
                        accs_tot_avg_km = {0.0001: [], 0.0005: [], 0.001: [], 0.005: [],
                                                0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}
                    for j in range(1, exec_n + 1, 1):
                        execute_kmeans(trainX, trainy, testX, testy, 300, v_avg_c, v_min_c, accs_tot_avg_km)
                    accs_tot_avg_min_km = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                        0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                    accs_tot_avg_mean_km = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                        0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                    accs_tot_avg_max_km = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0, 0.6: 0.0, 0.7: 0.0,
                                        0.8: 0.0, 0.9: 0.0, 1.0: 0.0}
                    if 'UCI' not in w_path:
                        accs_tot_avg_min_km = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                        accs_tot_avg_mean_km = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                        accs_tot_avg_max_km = {0.0001: 0.0, 0.0005: 0.0, 0.001: 0.0, 0.005: 0.0,
                                                0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0}
                    for k in accs_tot_avg_km.keys():
                        accs_tot_avg_mean_km.update({k: np.mean(accs_tot_avg_km[k])})
                        accs_tot_avg_max_km.update({k: np.max(accs_tot_avg_km[k])})
                        accs_tot_avg_min_km.update({k: np.min(accs_tot_avg_km[k])})
                    plot_som_comp(t_iter, accs_avg_mean, accs_min_mean, accs_avg_max, accs_min_max, accs_avg_min, accs_min_min, accs_tot_avg_mean_km, accs_tot_avg_min_km, accs_tot_avg_max_km)
                    count_anim += 1
        else:
            v_avg_c, v_min_c = get_anovaf(trainX, trainy, testX, testy)
            print()
            execute_kmeans(trainX, trainy, testX, testy, 300, v_avg_c, v_min_c)


# execute all
run_experiment(1)
