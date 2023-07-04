# A very simple Flask Hello World app for you to get started with...

import json
import pickle

import numpy as np
import scipy.signal as sp_sig
from flask import Flask, request
from scipy.stats import median_abs_deviation, entropy
from sklearn.decomposition import PCA

app = Flask(__name__)

def median_filter(input_signal, w_s):
    output_signal = np.zeros(input_signal.shape)
    for i in range(input_signal.size):
        start = max(0, i - w_s // 2)
        end = min(input_signal.size, i + w_s // 2 + 1)
        buffer = input_signal[start:end]
        median = np.median(buffer)
        output_signal[i] = median
    return output_signal

def thirdord_butt_filt(sig, sample_rate=41.29, cutoff_frequency=20.0):
    nyquist_rate = sample_rate / 2
    normal_cutoff = cutoff_frequency / nyquist_rate
    b, a = sp_sig.butter(3, normal_cutoff, btype='low', analog=False)
    filt_sig = sp_sig.filtfilt(b, a, sig)
    return filt_sig

def butterworth_coeffs(cutoff, o):
    coeffs = []
    for i in range(o + 1):
        coeffs.append(1.0 / (1.0 + np.power(cutoff, 2 * i)))
    return np.array(coeffs)

def butterworth_filter(data, cutoff, ordr):
    filtered_data = np.zeros(data.shape)
    coeffs = butterworth_coeffs(cutoff, ordr)
    for i in range(data.shape[0]):
        for j in range(ordr + 1):
            if i - j >= 0:
                filtered_data[i] += coeffs[j] * data[i - j]
    return filtered_data

# def butterworth_filter(sig, f_s, cf, order_):
#     nf = f_s / 2
#     normalised_corner_frequency = cf / nf
#     b, a = butter(order_, normalised_corner_frequency, btype='low', analog=True)
#     return filtfilt(b, a, sig)

def calculate_jerk(sig, f_s):
    # acceleration = np.gradient(sig, 1/f_s)
    # filtered_acceleration = savgol_filter(acceleration, window_size, polyorder)
    jerk = np.gradient(sig, 1/f_s, axis=1)
    return jerk

def calculate_magnitude(x, y, z):
    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    z2 = np.power(z, 2)
    # print(type(x2))
    # print(x2)
    # input()
    tmp_mag = np.add(x2, y2)
    # print(type(tmp))
    # print(tmp)
    # input()
    tmp_mag = np.add(tmp_mag, z2)
    # print(type(tmp))
    # print(tmp)
    # input()
    return np.sqrt(tmp_mag)

def calculate_sma(mag):
    n = len(mag)
    tmp_sma = np.sum(mag, axis=1)
    # print("tmp_sma: ", tmp_sma.shape)
    # input()
    return np.divide(tmp_sma, n)

def calculate_gravity(sig_tot, sig_body):
    return np.subtract(sig_tot, sig_body)

def calculate_energy(sig):
    # tmp_en = np.power(sig, 2)
    # print(tmp_en.shape)
    # input()
    tmp_en = np.sum(np.power(sig, 2), axis=1)
    # print(tmp_en.shape)
    # input()
    return np.divide(tmp_en, len(sig))

def calculate_iqr(sig):
    return np.percentile(sig, 75, axis=1) - np.percentile(sig, 25, axis=1)

def calculate_entropy(sig):
    # sig = np.array(sig)
    # print(np.min(sig))
    # print(np.max(sig))
    # print(sig.shape)
    # print(range(np.min(sig), np.max(sig)+2.0))
    # input()
    signal_hist, bin_edges = np.histogram(sig, bins=np.unique(sig))
    # print(signal_hist)
    # print(signal_hist.shape)
    # input()
    signal_hist = signal_hist / np.sum(signal_hist)
    return entropy(signal_hist)

def entropy_per_signal(sig):
    output = []
    for index, value in enumerate(sig):
        # print(calculate_entropy(value))
        # print(type(calculate_entropy(value)))
        en = calculate_entropy(value)
        output.append(en)
    return np.array(output)

def signal_autoregression_coefficients_burg(sig, ordr = 4):
    N = len(sig)
    a = np.zeros(ordr + 1)
    e = np.zeros(N)
    a[0] = 1
    for m in range(1, ordr + 1):
        num = 0
        den = 0
        for n in range(m, N):
            num += (sig[n] * sig[n - m])
            den += (sig[n] ** 2 + sig[n - m] ** 2)
        k = -2 * num / den
        a_temp = a.copy()
        a[m] = k * a[m - 1]
        for i in range(1, m):
            a[i] = a_temp[i] + k * a_temp[m - i]
    return a

def calculate_arcoeff(sig):
    output = []
    for ind, v in enumerate(sig):
        ar_c = signal_autoregression_coefficients_burg(v)
        output.append(ar_c[1:])
    return np.array(output)

def correlation_coefficient(signal1, signal2):
    n = len(signal1)
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    stddev1 = np.std(signal1)
    stddev2 = np.std(signal2)
    correlation = np.sum((signal1 - mean1) * (signal2 - mean2)) / (n * stddev1 * stddev2)
    return correlation

def calculate_correlation(sig1, sig2):
    output = []
    for ind, v in enumerate(sig1):
        corr = correlation_coefficient(v, sig2[ind])
        output.append(corr)
    return np.array(output)


@app.route('/som')
def test_som():
    ax = str(request.args.get('ax'))
    ay = str(request.args.get('ay'))
    az = str(request.args.get('az'))
    gx = str(request.args.get('gx'))
    gy = str(request.args.get('gy'))
    gz = str(request.args.get('gz'))
    ax = ax.split("_")
    ay = ay.split("_")
    az = az.split("_")
    gx = gx.split("_")
    gy = gy.split("_")
    gz = gz.split("_")
    trainX = np.dstack((ax, ay, az, gx, gy, gz))
    trainX = trainX.astype('float64')
    # NOISE FILTERING
    filtered_signals = np.empty((1, 128,))
    for idx, val in enumerate(trainX):
        filt_ax_val = np.empty((128,))
        for i in range(6):
            signal = val[:, i] # Your signal
            window_size = 64 # Window size for the median filter
            filtered_signal = thirdord_butt_filt(median_filter(signal, window_size))
            if i == 0:
                filt_ax_val = filtered_signal
            else:
                filt_ax_val = np.column_stack((filt_ax_val, filtered_signal))
        filt_ax_val = np.expand_dims(filt_ax_val, axis=0)
        if idx == 0:
            filtered_signals = filt_ax_val
        else:
            filtered_signals = np.concatenate((filtered_signals, filt_ax_val))

    # OBTAINING BODY AND GRAVITY ACC
    body_acc = np.empty((1, 128,))
    for idx, val in enumerate(filtered_signals):
        body_acc_val = np.empty((128,))
        for i in range(3):
            signal = val[:, i] # Your signal
            corner_frequency = 0.3 # Corner frequency for the Butterworth filter
            order = 2 # Order of the Butterworth filter
            filtered_signal = butterworth_filter(signal, corner_frequency, order)
            if i == 0:
                body_acc_val = filtered_signal
            else:
                body_acc_val = np.column_stack((body_acc_val, filtered_signal))
        body_acc_val = np.expand_dims(body_acc_val, axis=0)
        if idx == 0:
            body_acc = body_acc_val
        else:
            body_acc = np.concatenate((body_acc, body_acc_val))
    total_acc = np.empty((1, 128,))
    body_gyro = np.empty((1, 128,))
    for idx, val in enumerate(filtered_signals):
        tot_acc_val = np.empty((128,))
        body_gyro_val = np.empty((128,))
        for i in range(3):
            if i == 0:
                tot_acc_val = val[:, i]
                body_gyro_val = val[:, i + 3]
            else:
                tot_acc_val = np.column_stack((tot_acc_val, val[:, i]))
                body_gyro_val = np.column_stack((body_gyro_val, val[:, i + 3]))
        tot_acc_val = np.expand_dims(tot_acc_val, axis=0)
        body_gyro_val = np.expand_dims(body_gyro_val, axis=0)
        if idx == 0:
            total_acc = tot_acc_val
            body_gyro = body_gyro_val
        else:
            total_acc = np.concatenate((total_acc, tot_acc_val))
            body_gyro = np.concatenate((body_gyro, body_gyro_val))

    # JERK AND GRAVITY DATA COMPUTING
    tot_acc_x = total_acc[:, :, 0]
    tot_acc_y = total_acc[:, :, 1]
    tot_acc_z = total_acc[:, :, 2]
    body_acc_x = body_acc[:, :, 0]
    body_acc_y = body_acc[:, :, 1]
    body_acc_z = body_acc[:, :, 2]
    grav_acc_x = calculate_gravity(tot_acc_x, body_acc_x)
    grav_acc_y = calculate_gravity(tot_acc_y, body_acc_y)
    grav_acc_z = calculate_gravity(tot_acc_z, body_acc_z)
    body_gyro_x = body_gyro[:, :, 0]
    body_gyro_y = body_gyro[:, :, 1]
    body_gyro_z = body_gyro[:, :, 2]
    j_body_acc_x = calculate_jerk(body_acc_x, 41.29)
    j_body_acc_y = calculate_jerk(body_acc_y, 41.29)
    j_body_acc_z = calculate_jerk(body_acc_z, 41.29)
    j_body_gyro_x = calculate_jerk(body_gyro_x, 41.29)
    j_body_gyro_y = calculate_jerk(body_gyro_y, 41.29)
    j_body_gyro_z = calculate_jerk(body_gyro_z, 41.29)

    # MAG DATA COMPUTING
    mag_body_acc = calculate_magnitude(body_acc_x, body_acc_y, body_acc_z)
    mag_grav_acc = calculate_magnitude(grav_acc_x, grav_acc_y, grav_acc_z)
    mag_body_gyro = calculate_magnitude(body_gyro_x, body_gyro_y, body_gyro_z)
    mag_j_body_acc = calculate_magnitude(j_body_acc_x, j_body_acc_y, j_body_acc_z)
    mag_j_body_gyro = calculate_magnitude(j_body_gyro_x, j_body_gyro_y, j_body_gyro_z)

    feature_names = []
    # MEAN COMPUTING
    j_body_acc_x_mean = np.mean(j_body_acc_x, axis=1)
    features = j_body_acc_x_mean
    j_body_acc_z_mean = np.mean(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_mean))
    mag_body_gyro_mean = np.mean(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_mean))
    mag_j_body_acc_mean = np.mean(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_mean))
    feature_names.append(['tBodyAccJerk-mean()-X', 'tBodyAccJerk-mean()-Z']) # , 'tBodyAccJerk-mean()-Y'
    feature_names.append(['tBodyGyroMag-mean()', 'tBodyAccJerkMag-mean()']) # 'tGravityAccMag-mean()', 'tBodyAccMag-mean()', , 'tBodyGyroJerkMag-mean()'

    # std_dev COMPUTING
    grav_acc_x_std = np.std(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_std))
    grav_acc_y_std = np.std(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_std))
    grav_acc_z_std = np.std(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_std))
    body_acc_x_std = np.std(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_std))
    body_acc_y_std = np.std(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_std))
    body_acc_z_std = np.std(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_std))
    mag_grav_acc_std = np.std(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_std))
    mag_body_acc_std = np.std(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_std))
    mag_body_gyro_std = np.std(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_std))
    mag_j_body_acc_std = np.std(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_std))
    mag_j_body_gyro_std = np.std(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_std))
    feature_names.append(['tGravityAcc-std()-X', 'tGravityAcc-std()-Y', 'tGravityAcc-std()-Z'])
    feature_names.append(['tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z'])
    feature_names.append(['tGravityAccMag-std()', 'tBodyAccMag-std()', 'tBodyGyroMag-std()', 'tBodyAccJerkMag-std()', 'tBodyGyroJerkMag-std()'])

    # MEDIAN ABSOLUTE DEVIATION COMPUTING
    grav_acc_x_mad = median_abs_deviation(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_mad))
    grav_acc_y_mad = median_abs_deviation(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_mad))
    grav_acc_z_mad = median_abs_deviation(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_mad))
    body_acc_x_mad = median_abs_deviation(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_mad))
    body_acc_y_mad = median_abs_deviation(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_mad))
    body_acc_z_mad = median_abs_deviation(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_mad))
    j_body_acc_x_mad = median_abs_deviation(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_mad))
    j_body_acc_y_mad = median_abs_deviation(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_mad))
    j_body_acc_z_mad = median_abs_deviation(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_mad))
    mag_grav_acc_mad = median_abs_deviation(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_mad))
    mag_body_acc_mad = median_abs_deviation(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_mad))
    mag_j_body_acc_mad = median_abs_deviation(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_mad))
    mag_j_body_gyro_mad = median_abs_deviation(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_mad))
    feature_names.append(['tGravityAcc-mad()-X', 'tGravityAcc-mad()-Y', 'tGravityAcc-mad()-Z'])
    feature_names.append(['tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z'])
    feature_names.append(['tBodyAccJerk-mad()-X', 'tBodyAccJerk-mad()-Y', 'tBodyAccJerk-mad()-Z'])
    feature_names.append(['tGravityAccMag-mad()', 'tBodyAccMag-mad()', 'tBodyAccJerkMag-mad()', 'tBodyGyroJerkMag-mad()']) # , 'tBodyGyroMag-mad()'

    # MAX COMPUTING
    grav_acc_x_max = np.amax(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_max))
    grav_acc_z_max = np.amax(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_max))
    mag_j_body_gyro_max = np.amax(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_max))
    feature_names.append(['tGravityAcc-max()-X', 'tGravityAcc-max()-Z'])
    feature_names.append(['tBodyGyroJerkMag-max()']) # 'tGravityAccMag-max()', 'tBodyAccMag-max()', 'tBodyGyroMag-max()', 'tBodyAccJerkMag-max()',

    # MIN COMPUTING
    mag_grav_acc_min = np.amin(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_min))
    mag_j_body_acc_min = np.amin(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_min))
    feature_names.append(['tGravityAccMag-min()', 'tBodyAccJerkMag-min()']) # , 'tBodyAccMag-min()', 'tBodyGyroMag-min()', 'tBodyGyroJerkMag-min()'

    # SIGNAL MAGNITUDE AREA COMPUTING
    grav_acc_sma = calculate_sma(mag_grav_acc)
    features = np.column_stack((features, grav_acc_sma))
    body_acc_sma = calculate_sma(mag_body_acc)
    features = np.column_stack((features, body_acc_sma))
    body_gyro_sma = calculate_sma(mag_body_gyro)
    features = np.column_stack((features, body_gyro_sma))
    j_body_acc_sma = calculate_sma(mag_j_body_acc)
    features = np.column_stack((features, j_body_acc_sma))
    mag_grav_acc_sma = calculate_sma(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_sma))
    mag_body_acc_sma = calculate_sma(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_sma))
    mag_j_body_acc_sma = calculate_sma(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_sma))
    feature_names.append(['tGravityAcc-sma()', 'tBodyAcc-sma()', 'tBodyGyro-sma()', 'tBodyAccJerk-sma()']) # , 'tBodyGyroJerk-sma()'
    feature_names.append(['tGravityAccMag-sma()', 'tBodyAccMag-sma()', 'tBodyAccJerkMag-sma()']) # , 'tBodyGyroMag-sma()', 'tBodyGyroJerkMag-sma()'

    # ENERGY COMPUTING
    grav_acc_x_energy = calculate_energy(grav_acc_x)
    features = np.column_stack((features, grav_acc_x_energy))
    grav_acc_y_energy = calculate_energy(grav_acc_y)
    features = np.column_stack((features, grav_acc_y_energy))
    grav_acc_z_energy = calculate_energy(grav_acc_z)
    features = np.column_stack((features, grav_acc_z_energy))
    body_acc_x_energy = calculate_energy(body_acc_x)
    features = np.column_stack((features, body_acc_x_energy))
    body_acc_y_energy = calculate_energy(body_acc_y)
    features = np.column_stack((features, body_acc_y_energy))
    body_acc_z_energy = calculate_energy(body_acc_z)
    features = np.column_stack((features, body_acc_z_energy))
    j_body_acc_x_energy = calculate_energy(j_body_acc_x)
    features = np.column_stack((features, j_body_acc_x_energy))
    j_body_acc_y_energy = calculate_energy(j_body_acc_y)
    features = np.column_stack((features, j_body_acc_y_energy))
    j_body_acc_z_energy = calculate_energy(j_body_acc_z)
    features = np.column_stack((features, j_body_acc_z_energy))
    mag_grav_acc_energy = calculate_energy(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_energy))
    mag_body_acc_energy = calculate_energy(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_energy))
    mag_j_body_acc_energy = calculate_energy(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_energy))
    feature_names.append(['tGravityAcc-energy()-X', 'tGravityAcc-energy()-Y', 'tGravityAcc-energy()-Z'])
    feature_names.append(['tBodyAcc-energy()-X', 'tBodyAcc-energy()-Y', 'tBodyAcc-energy()-Z'])
    feature_names.append(['tBodyAccJerk-energy()-X', 'tBodyAccJerk-energy()-Y', 'tBodyAccJerk-energy()-Z'])
    feature_names.append(['tGravityAccMag-energy()', 'tBodyAccMag-energy()', 'tBodyAccJerkMag-energy()']) #, 'tBodyGyroMag-energy()', 'tBodyGyroJerkMag-energy()'

    # INTERQUARTILE RANGE COMPUTING
    grav_acc_x_iqr = calculate_iqr(grav_acc_x)
    features = np.column_stack((features, grav_acc_x_iqr))
    grav_acc_y_iqr = calculate_iqr(grav_acc_y)
    features = np.column_stack((features, grav_acc_y_iqr))
    grav_acc_z_iqr = calculate_iqr(grav_acc_z)
    features = np.column_stack((features, grav_acc_z_iqr))
    body_acc_x_iqr = calculate_iqr(body_acc_x)
    features = np.column_stack((features, body_acc_x_iqr))
    body_acc_y_iqr = calculate_iqr(body_acc_y)
    features = np.column_stack((features, body_acc_y_iqr))
    body_acc_z_iqr = calculate_iqr(body_acc_z)
    features = np.column_stack((features, body_acc_z_iqr))
    j_body_acc_z_iqr = calculate_iqr(j_body_acc_z)
    features = np.column_stack((features, j_body_acc_z_iqr))
    mag_grav_acc_iqr = calculate_iqr(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_iqr))
    mag_body_acc_iqr = calculate_iqr(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_iqr))
    feature_names.append(['tGravityAcc-iqr()-X', 'tGravityAcc-iqr()-Y', 'tGravityAcc-iqr()-Z'])
    feature_names.append(['tBodyAcc-iqr()-X', 'tBodyAcc-iqr()-Y', 'tBodyAcc-iqr()-Z'])
    feature_names.append(['tBodyAccJerk-iqr()-Z']) # 'tBodyAccJerk-iqr()-X', 'tBodyAccJerk-iqr()-Y',
    feature_names.append(['tGravityAccMag-iqr()', 'tBodyAccMag-iqr()']) # , 'tBodyGyroMag-iqr()', 'tBodyAccJerkMag-iqr()', 'tBodyGyroJerkMag-iqr()'

    # ENTROPY COMPUTING
    # print(entropy_per_signal(grav_acc_x))
    # grav_acc_x_entropy = entropy_per_signal(grav_acc_x)
    # features = np.column_stack((features, grav_acc_x_entropy))
    # grav_acc_y_entropy = entropy_per_signal(grav_acc_y)
    # features = np.column_stack((features, grav_acc_y_entropy))
    # grav_acc_z_entropy = entropy_per_signal(grav_acc_z)
    # features = np.column_stack((features, grav_acc_z_entropy))
    # body_acc_x_entropy = entropy_per_signal(body_acc_x)
    # features = np.column_stack((features, body_acc_x_entropy))
    # body_acc_y_entropy = entropy_per_signal(body_acc_y)
    # features = np.column_stack((features, body_acc_y_entropy))
    # body_acc_z_entropy = entropy_per_signal(body_acc_z)
    # features = np.column_stack((features, body_acc_z_entropy))
    # body_gyro_x_entropy = entropy_per_signal(body_gyro_x)
    # features = np.column_stack((features, body_gyro_x_entropy))
    # body_gyro_y_entropy = entropy_per_signal(body_gyro_y)
    # features = np.column_stack((features, body_gyro_y_entropy))
    # body_gyro_z_entropy = entropy_per_signal(body_gyro_z)
    # features = np.column_stack((features, body_gyro_z_entropy))
    # j_body_acc_x_entropy = entropy_per_signal(j_body_acc_x)
    # features = np.column_stack((features, j_body_acc_x_entropy))
    # j_body_acc_y_entropy = entropy_per_signal(j_body_acc_y)
    # features = np.column_stack((features, j_body_acc_y_entropy))
    # j_body_acc_z_entropy = entropy_per_signal(j_body_acc_z)
    # features = np.column_stack((features, j_body_acc_z_entropy))
    # j_body_gyro_x_entropy = entropy_per_signal(j_body_gyro_x)
    # features = np.column_stack((features, j_body_gyro_x_entropy))
    # j_body_gyro_y_entropy = entropy_per_signal(j_body_gyro_y)
    # features = np.column_stack((features, j_body_gyro_y_entropy))
    # j_body_gyro_z_entropy = entropy_per_signal(j_body_gyro_z)
    # features = np.column_stack((features, j_body_gyro_z_entropy))
    # mag_grav_acc_entropy = entropy_per_signal(mag_grav_acc)
    # features = np.column_stack((features, mag_grav_acc_entropy))
    # mag_body_acc_entropy = entropy_per_signal(mag_body_acc)
    # features = np.column_stack((features, mag_body_acc_entropy))
    # mag_body_gyro_entropy = entropy_per_signal(mag_body_gyro)
    # features = np.column_stack((features, mag_body_gyro_entropy))
    # mag_j_body_acc_entropy = entropy_per_signal(mag_j_body_acc)
    # features = np.column_stack((features, mag_j_body_acc_entropy))
    # mag_j_body_gyro_entropy = entropy_per_signal(mag_j_body_gyro)
    # features = np.column_stack((features, mag_j_body_gyro_entropy))
    # feature_names.append(['tGravityAcc-entropy()-X', 'tGravityAcc-entropy()-Y', 'tGravityAcc-entropy()-Z'])
    # feature_names.append(['tBodyAcc-entropy()-X', 'tBodyAcc-entropy()-Y', 'tBodyAcc-entropy()-Z'])
    # feature_names.append(['tBodyGyro-entropy()-X', 'tBodyGyro-entropy()-Y', 'tBodyGyro-entropy()-Z'])
    # feature_names.append(['tBodyAccJerk-entropy()-X', 'tBodyAccJerk-entropy()-Y', 'tBodyAccJerk-entropy()-Z'])
    # feature_names.append(['tBodyGyroJerk-entropy()-X', 'tBodyGyroJerk-entropy()-Y', 'tBodyGyroJerk-entropy()-Z'])
    # feature_names.append(['tGravityAccMag-entropy()', 'tBodyAccMag-entropy()', 'tBodyGyroMag-entropy()', 'tBodyAccJerkMag-entropy()', 'tBodyGyroJerkMag-entropy()'])

    # AR COEFFICIENTS COMPUTING
    grav_acc_x_arcoeff = calculate_arcoeff(grav_acc_x)
    for i in range(3):
        features = np.column_stack((features, grav_acc_x_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-X,' + str(i + 1))
    grav_acc_y_arcoeff = calculate_arcoeff(grav_acc_y)
    for i in range(3):
        features = np.column_stack((features, grav_acc_y_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-Y,' + str(i + 1))
    grav_acc_z_arcoeff = calculate_arcoeff(grav_acc_z)
    for i in range(3):
        features = np.column_stack((features, grav_acc_z_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-Z,' + str(i + 1))
    body_acc_x_arcoeff = calculate_arcoeff(body_acc_x)
    for i in range(3):
        features = np.column_stack((features, body_acc_x_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-X,' + str(i + 1))
    body_acc_y_arcoeff = calculate_arcoeff(body_acc_y)
    for i in range(3):
        features = np.column_stack((features, body_acc_y_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-Y,' + str(i + 1))
    body_acc_z_arcoeff = calculate_arcoeff(body_acc_z)
    for i in range(3):
        features = np.column_stack((features, body_acc_z_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-Z,' + str(i + 1))
    body_gyro_x_arcoeff = calculate_arcoeff(body_gyro_x)
    for i in range(3):
        features = np.column_stack((features, body_gyro_x_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-X,' + str(i + 1))
    body_gyro_y_arcoeff = calculate_arcoeff(body_gyro_y)
    for i in range(3):
        features = np.column_stack((features, body_gyro_y_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-Y,' + str(i + 1))
    body_gyro_z_arcoeff = calculate_arcoeff(body_gyro_z)
    for i in range(3):
        features = np.column_stack((features, body_gyro_z_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-Z,' + str(i + 1))
    j_body_acc_x_arcoeff = calculate_arcoeff(j_body_acc_x)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_acc_x_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-X,' + str(i + 1))
    j_body_acc_y_arcoeff = calculate_arcoeff(j_body_acc_y)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_acc_y_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-Y,' + str(i + 1))
    j_body_acc_z_arcoeff = calculate_arcoeff(j_body_acc_z)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_acc_z_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-Z,' + str(i + 1))
    j_body_gyro_x_arcoeff = calculate_arcoeff(j_body_gyro_x)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_gyro_x_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-X,' + str(i + 1))
    j_body_gyro_y_arcoeff = calculate_arcoeff(j_body_gyro_y)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_gyro_y_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-Y,' + str(i + 1))
    j_body_gyro_z_arcoeff = calculate_arcoeff(j_body_gyro_z)
    for i in range(1, 4):
        features = np.column_stack((features, j_body_gyro_z_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-Z,' + str(i + 1))
    mag_grav_acc_arcoeff = calculate_arcoeff(mag_grav_acc)
    for i in range(3):
        features = np.column_stack((features, mag_grav_acc_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-X,' + str(i + 1))
    mag_body_acc_arcoeff = calculate_arcoeff(mag_body_acc)
    for i in range(1, 4):
        features = np.column_stack((features, mag_body_acc_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-Y,' + str(i + 1))
    mag_body_gyro_arcoeff = calculate_arcoeff(mag_body_gyro)
    for i in range(1):
        features = np.column_stack((features, mag_body_gyro_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-Z,' + str(i + 1))
    mag_j_body_acc_arcoeff = calculate_arcoeff(mag_j_body_acc)
    for i in range(1, 4, 2):
        features = np.column_stack((features, mag_j_body_acc_arcoeff[:, i]))
        feature_names.append('tBodyAccMag-arCoeff()-X,' + str(i + 1))

    # CORRELATION COMPUTING
    j_body_gyro_xy_corr = calculate_correlation(j_body_gyro_x, j_body_gyro_y)
    features = np.column_stack((features, j_body_gyro_xy_corr))
    j_body_gyro_xz_corr = calculate_correlation(j_body_gyro_x, j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_xz_corr))
    j_body_gyro_yz_corr = calculate_correlation(j_body_gyro_y, j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_yz_corr))
    feature_names.append(['tBodyGyroJerk-correlation()-X,Y', 'tBodyGyroJerk-correlation()-X,Z', 'tBodyGyroJerk-correlation()-Y,Z'])
    feature_names_def = []
    for idx, val in enumerate(feature_names):
        if len(val) <= 5:
            for idx2, val2, in enumerate(feature_names[idx]):
                if val2 != '':
                    feature_names_def.append(val2)
        else:
            feature_names_def.append(val)

    features_ordered = np.empty((features.shape[0],))
    to_overwrite = True
    feature_names_def_ordered = []

    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAcc':
            if to_overwrite:
                features_ordered = features[:, idx]
                to_overwrite = False
            else:
                features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tGravityAcc':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccJerk':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyro':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroJerk':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tGravityAccMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccJerkMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroJerkMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    to_ret = ""
    for idx, val in enumerate(features_ordered):
        for idx2, val2, in enumerate(features_ordered[idx]):
            to_ret += str(val2) + ' '
    return str(to_ret)

@app.route('/kmeans')
def test_kmeans():
    ax = str(request.args.get('ax'))
    ay = str(request.args.get('ay'))
    az = str(request.args.get('az'))
    gx = str(request.args.get('gx'))
    gy = str(request.args.get('gy'))
    gz = str(request.args.get('gz'))
    ax = ax.split("_")
    ay = ay.split("_")
    az = az.split("_")
    gx = gx.split("_")
    gy = gy.split("_")
    gz = gz.split("_")
    trainX = np.dstack((ax, ay, az, gx, gy, gz))
    trainX = trainX.astype('float64')
    # NOISE FILTERING
    filtered_signals = np.empty((1, 128,))
    for idx, val in enumerate(trainX):
        filt_ax_val = np.empty((128,))
        for i in range(6):
            signal = val[:, i] # Your signal
            window_size = 64 # Window size for the median filter
            filtered_signal = thirdord_butt_filt(median_filter(signal, window_size))
            if i == 0:
                filt_ax_val = filtered_signal
            else:
                filt_ax_val = np.column_stack((filt_ax_val, filtered_signal))
        filt_ax_val = np.expand_dims(filt_ax_val, axis=0)
        if idx == 0:
            filtered_signals = filt_ax_val
        else:
            filtered_signals = np.concatenate((filtered_signals, filt_ax_val))

    # OBTAINING BODY AND GRAVITY ACC
    body_acc = np.empty((1, 128,))
    for idx, val in enumerate(filtered_signals):
        body_acc_val = np.empty((128,))
        for i in range(3):
            signal = val[:, i] # Your signal
            corner_frequency = 0.3 # Corner frequency for the Butterworth filter
            order = 2 # Order of the Butterworth filter

            filtered_signal = butterworth_filter(signal, corner_frequency, order)
            if i == 0:
                body_acc_val = filtered_signal
            else:
                body_acc_val = np.column_stack((body_acc_val, filtered_signal))
        body_acc_val = np.expand_dims(body_acc_val, axis=0)
        if idx == 0:
            body_acc = body_acc_val
        else:
            body_acc = np.concatenate((body_acc, body_acc_val))

    total_acc = np.empty((1, 128,))
    body_gyro = np.empty((1, 128,))
    for idx, val in enumerate(filtered_signals):
        tot_acc_val = np.empty((128,))
        body_gyro_val = np.empty((128,))
        for i in range(3):
            if i == 0:
                tot_acc_val = val[:, i]
                body_gyro_val = val[:, i + 3]
            else:
                tot_acc_val = np.column_stack((tot_acc_val, val[:, i]))
                body_gyro_val = np.column_stack((body_gyro_val, val[:, i + 3]))
        tot_acc_val = np.expand_dims(tot_acc_val, axis=0)
        body_gyro_val = np.expand_dims(body_gyro_val, axis=0)
        if idx == 0:
            total_acc = tot_acc_val
            body_gyro = body_gyro_val
        else:
            total_acc = np.concatenate((total_acc, tot_acc_val))
            body_gyro = np.concatenate((body_gyro, body_gyro_val))

    # JERK AND GRAVITY DATA COMPUTING
    tot_acc_x = total_acc[:, :, 0]
    tot_acc_y = total_acc[:, :, 1]
    tot_acc_z = total_acc[:, :, 2]
    body_acc_x = body_acc[:, :, 0]
    body_acc_y = body_acc[:, :, 1]
    body_acc_z = body_acc[:, :, 2]
    grav_acc_x = calculate_gravity(tot_acc_x, body_acc_x)
    grav_acc_y = calculate_gravity(tot_acc_y, body_acc_y)
    grav_acc_z = calculate_gravity(tot_acc_z, body_acc_z)
    body_gyro_x = body_gyro[:, :, 0]
    body_gyro_y = body_gyro[:, :, 1]
    body_gyro_z = body_gyro[:, :, 2]
    j_body_acc_x = calculate_jerk(body_acc_x, 41.29)
    j_body_acc_y = calculate_jerk(body_acc_y, 41.29)
    j_body_acc_z = calculate_jerk(body_acc_z, 41.29)
    j_body_gyro_x = calculate_jerk(body_gyro_x, 41.29)
    j_body_gyro_y = calculate_jerk(body_gyro_y, 41.29)
    j_body_gyro_z = calculate_jerk(body_gyro_z, 41.29)

    # MAG DATA COMPUTING
    mag_body_acc = calculate_magnitude(body_acc_x, body_acc_y, body_acc_z)
    mag_grav_acc = calculate_magnitude(grav_acc_x, grav_acc_y, grav_acc_z)
    mag_body_gyro = calculate_magnitude(body_gyro_x, body_gyro_y, body_gyro_z)
    mag_j_body_acc = calculate_magnitude(j_body_acc_x, j_body_acc_y, j_body_acc_z)
    mag_j_body_gyro = calculate_magnitude(j_body_gyro_x, j_body_gyro_y, j_body_gyro_z)

    feature_names = []
    # MEAN COMPUTING
    grav_acc_x_mean = np.mean(grav_acc_x, axis=1)
    features = grav_acc_x_mean
    grav_acc_y_mean = np.mean(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_mean))
    grav_acc_z_mean = np.mean(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_mean))
    body_acc_x_mean = np.mean(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_mean))
    body_acc_y_mean = np.mean(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_mean))
    body_acc_z_mean = np.mean(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_mean))
    body_gyro_x_mean = np.mean(body_gyro_x, axis=1)
    features = np.column_stack((features, body_gyro_x_mean))
    body_gyro_y_mean = np.mean(body_gyro_y, axis=1)
    features = np.column_stack((features, body_gyro_y_mean))
    body_gyro_z_mean = np.mean(body_gyro_z, axis=1)
    features = np.column_stack((features, body_gyro_z_mean))
    j_body_acc_x_mean = np.mean(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_mean))
    j_body_acc_y_mean = np.mean(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_mean))
    j_body_acc_z_mean = np.mean(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_mean))
    j_body_gyro_x_mean = np.mean(j_body_gyro_x, axis=1)
    features = np.column_stack((features, j_body_gyro_x_mean))
    j_body_gyro_y_mean = np.mean(j_body_gyro_y, axis=1)
    features = np.column_stack((features, j_body_gyro_y_mean))
    j_body_gyro_z_mean = np.mean(j_body_gyro_z, axis=1)
    features = np.column_stack((features, j_body_gyro_z_mean))
    mag_grav_acc_mean = np.mean(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_mean))
    mag_body_acc_mean = np.mean(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_mean))
    mag_body_gyro_mean = np.mean(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_mean))
    mag_j_body_acc_mean = np.mean(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_mean))
    mag_j_body_gyro_mean = np.mean(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_mean))
    feature_names.append(['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-mean()-Z'])
    feature_names.append(['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z'])
    feature_names.append(['tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z'])
    feature_names.append(['tBodyAccJerk-mean()-X', 'tBodyAccJerk-mean()-Y', 'tBodyAccJerk-mean()-Z'])
    feature_names.append(['tBodyGyroJerk-mean()-X', 'tBodyGyroJerk-mean()-Y', 'tBodyGyroJerk-mean()-Z'])
    feature_names.append(['tGravityAccMag-mean()', 'tBodyAccMag-mean()', 'tBodyGyroMag-mean()', 'tBodyAccJerkMag-mean()', 'tBodyGyroJerkMag-mean()'])
    # std_dev COMPUTING
    grav_acc_x_std = np.std(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_std))
    grav_acc_y_std = np.std(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_std))
    grav_acc_z_std = np.std(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_std))
    body_acc_x_std = np.std(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_std))
    body_acc_y_std = np.std(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_std))
    body_acc_z_std = np.std(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_std))
    body_gyro_x_std = np.std(body_gyro_x, axis=1)
    features = np.column_stack((features, body_gyro_x_std))
    body_gyro_y_std = np.std(body_gyro_y, axis=1)
    features = np.column_stack((features, body_gyro_y_std))
    body_gyro_z_std = np.std(body_gyro_z, axis=1)
    features = np.column_stack((features, body_gyro_z_std))
    j_body_acc_x_std = np.std(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_std))
    j_body_acc_y_std = np.std(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_std))
    j_body_acc_z_std = np.std(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_std))
    j_body_gyro_x_std = np.std(j_body_gyro_x, axis=1)
    features = np.column_stack((features, j_body_gyro_x_std))
    j_body_gyro_y_std = np.std(j_body_gyro_y, axis=1)
    features = np.column_stack((features, j_body_gyro_y_std))
    j_body_gyro_z_std = np.std(j_body_gyro_z, axis=1)
    features = np.column_stack((features, j_body_gyro_z_std))
    mag_grav_acc_std = np.std(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_std))
    mag_body_acc_std = np.std(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_std))
    mag_body_gyro_std = np.std(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_std))
    mag_j_body_acc_std = np.std(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_std))
    mag_j_body_gyro_std = np.std(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_std))
    feature_names.append(['tGravityAcc-std()-X', 'tGravityAcc-std()-Y', 'tGravityAcc-std()-Z'])
    feature_names.append(['tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z'])
    feature_names.append(['tBodyGyro-std()-X', 'tBodyGyro-std()-Y', 'tBodyGyro-std()-Z'])
    feature_names.append(['tBodyAccJerk-std()-X', 'tBodyAccJerk-std()-Y', 'tBodyAccJerk-std()-Z'])
    feature_names.append(['tBodyGyroJerk-std()-X', 'tBodyGyroJerk-std()-Y', 'tBodyGyroJerk-std()-Z'])
    feature_names.append(['tGravityAccMag-std()', 'tBodyAccMag-std()', 'tBodyGyroMag-std()', 'tBodyAccJerkMag-std()', 'tBodyGyroJerkMag-std()'])

    # MEDIAN ABSOLUTE DEVIATION COMPUTING
    grav_acc_x_mad = median_abs_deviation(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_mad))
    grav_acc_y_mad = median_abs_deviation(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_mad))
    grav_acc_z_mad = median_abs_deviation(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_mad))
    body_acc_x_mad = median_abs_deviation(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_mad))
    body_acc_y_mad = median_abs_deviation(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_mad))
    body_acc_z_mad = median_abs_deviation(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_mad))
    body_gyro_x_mad = median_abs_deviation(body_gyro_x, axis=1)
    features = np.column_stack((features, body_gyro_x_mad))
    body_gyro_y_mad = median_abs_deviation(body_gyro_y, axis=1)
    features = np.column_stack((features, body_gyro_y_mad))
    body_gyro_z_mad = median_abs_deviation(body_gyro_z, axis=1)
    features = np.column_stack((features, body_gyro_z_mad))
    j_body_acc_x_mad = median_abs_deviation(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_mad))
    j_body_acc_y_mad = median_abs_deviation(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_mad))
    j_body_acc_z_mad = median_abs_deviation(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_mad))
    j_body_gyro_x_mad = median_abs_deviation(j_body_gyro_x, axis=1)
    features = np.column_stack((features, j_body_gyro_x_mad))
    j_body_gyro_y_mad = median_abs_deviation(j_body_gyro_y, axis=1)
    features = np.column_stack((features, j_body_gyro_y_mad))
    j_body_gyro_z_mad = median_abs_deviation(j_body_gyro_z, axis=1)
    features = np.column_stack((features, j_body_gyro_z_mad))
    mag_grav_acc_mad = median_abs_deviation(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_mad))
    mag_body_acc_mad = median_abs_deviation(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_mad))
    mag_body_gyro_mad = median_abs_deviation(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_mad))
    mag_j_body_acc_mad = median_abs_deviation(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_mad))
    mag_j_body_gyro_mad = median_abs_deviation(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_mad))
    feature_names.append(['tGravityAcc-mad()-X', 'tGravityAcc-mad()-Y', 'tGravityAcc-mad()-Z'])
    feature_names.append(['tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z'])
    feature_names.append(['tBodyGyro-mad()-X', 'tBodyGyro-mad()-Y', 'tBodyGyro-mad()-Z'])
    feature_names.append(['tBodyAccJerk-mad()-X', 'tBodyAccJerk-mad()-Y', 'tBodyAccJerk-mad()-Z'])
    feature_names.append(['tBodyGyroJerk-mad()-X', 'tBodyGyroJerk-mad()-Y', 'tBodyGyroJerk-mad()-Z'])
    feature_names.append(['tGravityAccMag-mad()', 'tBodyAccMag-mad()', 'tBodyGyroMag-mad()', 'tBodyAccJerkMag-mad()', 'tBodyGyroJerkMag-mad()'])

    # MAX COMPUTING
    grav_acc_x_max = np.amax(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_max))
    grav_acc_y_max = np.amax(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_max))
    grav_acc_z_max = np.amax(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_max))
    body_acc_x_max = np.amax(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_max))
    body_acc_y_max = np.amax(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_max))
    body_acc_z_max = np.amax(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_max))
    body_gyro_x_max = np.amax(body_gyro_x, axis=1)
    features = np.column_stack((features, body_gyro_x_max))
    body_gyro_y_max = np.amax(body_gyro_y, axis=1)
    features = np.column_stack((features, body_gyro_y_max))
    body_gyro_z_max = np.amax(body_gyro_z, axis=1)
    features = np.column_stack((features, body_gyro_z_max))
    j_body_acc_x_max = np.amax(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_max))
    j_body_acc_y_max = np.amax(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_max))
    j_body_acc_z_max = np.amax(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_max))
    j_body_gyro_x_max = np.amax(j_body_gyro_x, axis=1)
    features = np.column_stack((features, j_body_gyro_x_max))
    j_body_gyro_y_max = np.amax(j_body_gyro_y, axis=1)
    features = np.column_stack((features, j_body_gyro_y_max))
    j_body_gyro_z_max = np.amax(j_body_gyro_z, axis=1)
    features = np.column_stack((features, j_body_gyro_z_max))
    mag_grav_acc_max = np.amax(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_max))
    mag_body_acc_max = np.amax(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_max))
    mag_body_gyro_max = np.amax(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_max))
    mag_j_body_acc_max = np.amax(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_max))
    mag_j_body_gyro_max = np.amax(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_max))
    feature_names.append(['tGravityAcc-max()-X', 'tGravityAcc-max()-Y', 'tGravityAcc-max()-Z'])
    feature_names.append(['tBodyAcc-max()-X', 'tBodyAcc-max()-Y', 'tBodyAcc-max()-Z'])
    feature_names.append(['tBodyGyro-max()-X', 'tBodyGyro-max()-Y', 'tBodyGyro-max()-Z'])
    feature_names.append(['tBodyAccJerk-max()-X', 'tBodyAccJerk-max()-Y', 'tBodyAccJerk-max()-Z'])
    feature_names.append(['tBodyGyroJerk-max()-X', 'tBodyGyroJerk-max()-Y', 'tBodyGyroJerk-max()-Z'])
    feature_names.append(['tGravityAccMag-max()', 'tBodyAccMag-max()', 'tBodyGyroMag-max()', 'tBodyAccJerkMag-max()', 'tBodyGyroJerkMag-max()'])

    # MIN COMPUTING
    grav_acc_x_min = np.amin(grav_acc_x, axis=1)
    features = np.column_stack((features, grav_acc_x_min))
    grav_acc_y_min = np.amin(grav_acc_y, axis=1)
    features = np.column_stack((features, grav_acc_y_min))
    grav_acc_z_min = np.amin(grav_acc_z, axis=1)
    features = np.column_stack((features, grav_acc_z_min))
    body_acc_x_min = np.amin(body_acc_x, axis=1)
    features = np.column_stack((features, body_acc_x_min))
    body_acc_y_min = np.amin(body_acc_y, axis=1)
    features = np.column_stack((features, body_acc_y_min))
    body_acc_z_min = np.amin(body_acc_z, axis=1)
    features = np.column_stack((features, body_acc_z_min))
    body_gyro_x_min = np.amin(body_gyro_x, axis=1)
    features = np.column_stack((features, body_gyro_x_min))
    body_gyro_y_min = np.amin(body_gyro_y, axis=1)
    features = np.column_stack((features, body_gyro_y_min))
    body_gyro_z_min = np.amin(body_gyro_z, axis=1)
    features = np.column_stack((features, body_gyro_z_min))
    j_body_acc_x_min = np.amin(j_body_acc_x, axis=1)
    features = np.column_stack((features, j_body_acc_x_min))
    j_body_acc_y_min = np.amin(j_body_acc_y, axis=1)
    features = np.column_stack((features, j_body_acc_y_min))
    j_body_acc_z_min = np.amin(j_body_acc_z, axis=1)
    features = np.column_stack((features, j_body_acc_z_min))
    j_body_gyro_x_min = np.amin(j_body_gyro_x, axis=1)
    features = np.column_stack((features, j_body_gyro_x_min))
    j_body_gyro_y_min = np.amin(j_body_gyro_y, axis=1)
    features = np.column_stack((features, j_body_gyro_y_min))
    j_body_gyro_z_min = np.amin(j_body_gyro_z, axis=1)
    features = np.column_stack((features, j_body_gyro_z_min))
    mag_grav_acc_min = np.amin(mag_grav_acc, axis=1)
    features = np.column_stack((features, mag_grav_acc_min))
    mag_body_acc_min = np.amin(mag_body_acc, axis=1)
    features = np.column_stack((features, mag_body_acc_min))
    mag_body_gyro_min = np.amin(mag_body_gyro, axis=1)
    features = np.column_stack((features, mag_body_gyro_min))
    mag_j_body_acc_min = np.amin(mag_j_body_acc, axis=1)
    features = np.column_stack((features, mag_j_body_acc_min))
    mag_j_body_gyro_min = np.amin(mag_j_body_gyro, axis=1)
    features = np.column_stack((features, mag_j_body_gyro_min))
    feature_names.append(['tGravityAcc-min()-X', 'tGravityAcc-min()-Y', 'tGravityAcc-min()-Z'])
    feature_names.append(['tBodyAcc-min()-X', 'tBodyAcc-min()-Y', 'tBodyAcc-min()-Z'])
    feature_names.append(['tBodyGyro-min()-X', 'tBodyGyro-min()-Y', 'tBodyGyro-min()-Z'])
    feature_names.append(['tBodyAccJerk-min()-X', 'tBodyAccJerk-min()-Y', 'tBodyAccJerk-min()-Z'])
    feature_names.append(['tBodyGyroJerk-min()-X', 'tBodyGyroJerk-min()-Y', 'tBodyGyroJerk-min()-Z'])
    feature_names.append(['tGravityAccMag-min()', 'tBodyAccMag-min()', 'tBodyGyroMag-min()', 'tBodyAccJerkMag-min()', 'tBodyGyroJerkMag-min()'])
    # print(features.shape)

    # SIGNAL MAGNITUDE AREA COMPUTING
    grav_acc_sma = calculate_sma(mag_grav_acc)
    features = np.column_stack((features, grav_acc_sma))
    body_acc_sma = calculate_sma(mag_body_acc)
    features = np.column_stack((features, body_acc_sma))
    body_gyro_sma = calculate_sma(mag_body_gyro)
    features = np.column_stack((features, body_gyro_sma))
    j_body_acc_sma = calculate_sma(mag_j_body_acc)
    features = np.column_stack((features, j_body_acc_sma))
    j_body_gyro_sma = calculate_sma(mag_j_body_gyro)
    features = np.column_stack((features, j_body_gyro_sma))
    # forse da non considerare
    mag_grav_acc_sma = calculate_sma(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_sma))
    mag_body_acc_sma = calculate_sma(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_sma))
    mag_body_gyro_sma = calculate_sma(mag_body_gyro)
    features = np.column_stack((features, mag_body_gyro_sma))
    mag_j_body_acc_sma = calculate_sma(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_sma))
    mag_j_body_gyro_sma = calculate_sma(mag_j_body_gyro)
    features = np.column_stack((features, mag_j_body_gyro_sma))
    feature_names.append(['tGravityAcc-sma()', 'tBodyAcc-sma()', 'tBodyGyro-sma()', 'tBodyAccJerk-sma()', 'tBodyGyroJerk-sma()'])
    feature_names.append(['tGravityAccMag-sma()', 'tBodyAccMag-sma()', 'tBodyGyroMag-sma()', 'tBodyAccJerkMag-sma()', 'tBodyGyroJerkMag-sma()'])

    # ENERGY COMPUTING
    # print(calculate_energy(grav_acc_x))
    grav_acc_x_energy = calculate_energy(grav_acc_x)
    features = np.column_stack((features, grav_acc_x_energy))
    grav_acc_y_energy = calculate_energy(grav_acc_y)
    features = np.column_stack((features, grav_acc_y_energy))
    grav_acc_z_energy = calculate_energy(grav_acc_z)
    features = np.column_stack((features, grav_acc_z_energy))
    body_acc_x_energy = calculate_energy(body_acc_x)
    features = np.column_stack((features, body_acc_x_energy))
    body_acc_y_energy = calculate_energy(body_acc_y)
    features = np.column_stack((features, body_acc_y_energy))
    body_acc_z_energy = calculate_energy(body_acc_z)
    features = np.column_stack((features, body_acc_z_energy))
    body_gyro_x_energy = calculate_energy(body_gyro_x)
    features = np.column_stack((features, body_gyro_x_energy))
    body_gyro_y_energy = calculate_energy(body_gyro_y)
    features = np.column_stack((features, body_gyro_y_energy))
    body_gyro_z_energy = calculate_energy(body_gyro_z)
    features = np.column_stack((features, body_gyro_z_energy))
    j_body_acc_x_energy = calculate_energy(j_body_acc_x)
    features = np.column_stack((features, j_body_acc_x_energy))
    j_body_acc_y_energy = calculate_energy(j_body_acc_y)
    features = np.column_stack((features, j_body_acc_y_energy))
    j_body_acc_z_energy = calculate_energy(j_body_acc_z)
    features = np.column_stack((features, j_body_acc_z_energy))
    j_body_gyro_x_energy = calculate_energy(j_body_gyro_x)
    features = np.column_stack((features, j_body_gyro_x_energy))
    j_body_gyro_y_energy = calculate_energy(j_body_gyro_y)
    features = np.column_stack((features, j_body_gyro_y_energy))
    j_body_gyro_z_energy = calculate_energy(j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_z_energy))
    mag_grav_acc_energy = calculate_energy(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_energy))
    mag_body_acc_energy = calculate_energy(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_energy))
    mag_body_gyro_energy = calculate_energy(mag_body_gyro)
    features = np.column_stack((features, mag_body_gyro_energy))
    mag_j_body_acc_energy = calculate_energy(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_energy))
    mag_j_body_gyro_energy = calculate_energy(mag_j_body_gyro)
    features = np.column_stack((features, mag_j_body_gyro_energy))
    feature_names.append(['tGravityAcc-energy()-X', 'tGravityAcc-energy()-Y', 'tGravityAcc-energy()-Z'])
    feature_names.append(['tBodyAcc-energy()-X', 'tBodyAcc-energy()-Y', 'tBodyAcc-energy()-Z'])
    feature_names.append(['tBodyGyro-energy()-X', 'tBodyGyro-energy()-Y', 'tBodyGyro-energy()-Z'])
    feature_names.append(['tBodyAccJerk-energy()-X', 'tBodyAccJerk-energy()-Y', 'tBodyAccJerk-energy()-Z'])
    feature_names.append(['tBodyGyroJerk-energy()-X', 'tBodyGyroJerk-energy()-Y', 'tBodyGyroJerk-energy()-Z'])
    feature_names.append(['tGravityAccMag-energy()', 'tBodyAccMag-energy()', 'tBodyGyroMag-energy()', 'tBodyAccJerkMag-energy()', 'tBodyGyroJerkMag-energy()'])

    # INTERQUARTILE RANGE COMPUTING
    grav_acc_x_iqr = calculate_iqr(grav_acc_x)
    features = np.column_stack((features, grav_acc_x_iqr))
    grav_acc_y_iqr = calculate_iqr(grav_acc_y)
    features = np.column_stack((features, grav_acc_y_iqr))
    grav_acc_z_iqr = calculate_iqr(grav_acc_z)
    features = np.column_stack((features, grav_acc_z_iqr))
    body_acc_x_iqr = calculate_iqr(body_acc_x)
    features = np.column_stack((features, body_acc_x_iqr))
    body_acc_y_iqr = calculate_iqr(body_acc_y)
    features = np.column_stack((features, body_acc_y_iqr))
    body_acc_z_iqr = calculate_iqr(body_acc_z)
    features = np.column_stack((features, body_acc_z_iqr))
    body_gyro_x_iqr = calculate_iqr(body_gyro_x)
    features = np.column_stack((features, body_gyro_x_iqr))
    body_gyro_y_iqr = calculate_iqr(body_gyro_y)
    features = np.column_stack((features, body_gyro_y_iqr))
    body_gyro_z_iqr = calculate_iqr(body_gyro_z)
    features = np.column_stack((features, body_gyro_z_iqr))
    j_body_acc_x_iqr = calculate_iqr(j_body_acc_x)
    features = np.column_stack((features, j_body_acc_x_iqr))
    j_body_acc_y_iqr = calculate_iqr(j_body_acc_y)
    features = np.column_stack((features, j_body_acc_y_iqr))
    j_body_acc_z_iqr = calculate_iqr(j_body_acc_z)
    features = np.column_stack((features, j_body_acc_z_iqr))
    j_body_gyro_x_iqr = calculate_iqr(j_body_gyro_x)
    features = np.column_stack((features, j_body_gyro_x_iqr))
    j_body_gyro_y_iqr = calculate_iqr(j_body_gyro_y)
    features = np.column_stack((features, j_body_gyro_y_iqr))
    j_body_gyro_z_iqr = calculate_iqr(j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_z_iqr))
    mag_grav_acc_iqr = calculate_iqr(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_iqr))
    mag_body_acc_iqr = calculate_iqr(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_iqr))
    mag_body_gyro_iqr = calculate_iqr(mag_body_gyro)
    features = np.column_stack((features, mag_body_gyro_iqr))
    mag_j_body_acc_iqr = calculate_iqr(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_iqr))
    mag_j_body_gyro_iqr = calculate_iqr(mag_j_body_gyro)
    features = np.column_stack((features, mag_j_body_gyro_iqr))
    feature_names.append(['tGravityAcc-iqr()-X', 'tGravityAcc-iqr()-Y', 'tGravityAcc-iqr()-Z'])
    feature_names.append(['tBodyAcc-iqr()-X', 'tBodyAcc-iqr()-Y', 'tBodyAcc-iqr()-Z'])
    feature_names.append(['tBodyGyro-iqr()-X', 'tBodyGyro-iqr()-Y', 'tBodyGyro-iqr()-Z'])
    feature_names.append(['tBodyAccJerk-iqr()-X', 'tBodyAccJerk-iqr()-Y', 'tBodyAccJerk-iqr()-Z'])
    feature_names.append(['tBodyGyroJerk-iqr()-X', 'tBodyGyroJerk-iqr()-Y', 'tBodyGyroJerk-iqr()-Z'])
    feature_names.append(['tGravityAccMag-iqr()', 'tBodyAccMag-iqr()', 'tBodyGyroMag-iqr()', 'tBodyAccJerkMag-iqr()', 'tBodyGyroJerkMag-iqr()'])

    # ENTROPY COMPUTING
    # print(entropy_per_signal(grav_acc_x))
    grav_acc_x_entropy = entropy_per_signal(grav_acc_x)
    features = np.column_stack((features, grav_acc_x_entropy))
    grav_acc_y_entropy = entropy_per_signal(grav_acc_y)
    features = np.column_stack((features, grav_acc_y_entropy))
    grav_acc_z_entropy = entropy_per_signal(grav_acc_z)
    features = np.column_stack((features, grav_acc_z_entropy))
    body_acc_x_entropy = entropy_per_signal(body_acc_x)
    features = np.column_stack((features, body_acc_x_entropy))
    body_acc_y_entropy = entropy_per_signal(body_acc_y)
    features = np.column_stack((features, body_acc_y_entropy))
    body_acc_z_entropy = entropy_per_signal(body_acc_z)
    features = np.column_stack((features, body_acc_z_entropy))
    body_gyro_x_entropy = entropy_per_signal(body_gyro_x)
    features = np.column_stack((features, body_gyro_x_entropy))
    body_gyro_y_entropy = entropy_per_signal(body_gyro_y)
    features = np.column_stack((features, body_gyro_y_entropy))
    body_gyro_z_entropy = entropy_per_signal(body_gyro_z)
    features = np.column_stack((features, body_gyro_z_entropy))
    j_body_acc_x_entropy = entropy_per_signal(j_body_acc_x)
    features = np.column_stack((features, j_body_acc_x_entropy))
    j_body_acc_y_entropy = entropy_per_signal(j_body_acc_y)
    features = np.column_stack((features, j_body_acc_y_entropy))
    j_body_acc_z_entropy = entropy_per_signal(j_body_acc_z)
    features = np.column_stack((features, j_body_acc_z_entropy))
    j_body_gyro_x_entropy = entropy_per_signal(j_body_gyro_x)
    features = np.column_stack((features, j_body_gyro_x_entropy))
    j_body_gyro_y_entropy = entropy_per_signal(j_body_gyro_y)
    features = np.column_stack((features, j_body_gyro_y_entropy))
    j_body_gyro_z_entropy = entropy_per_signal(j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_z_entropy))
    mag_grav_acc_entropy = entropy_per_signal(mag_grav_acc)
    features = np.column_stack((features, mag_grav_acc_entropy))
    mag_body_acc_entropy = entropy_per_signal(mag_body_acc)
    features = np.column_stack((features, mag_body_acc_entropy))
    mag_body_gyro_entropy = entropy_per_signal(mag_body_gyro)
    features = np.column_stack((features, mag_body_gyro_entropy))
    mag_j_body_acc_entropy = entropy_per_signal(mag_j_body_acc)
    features = np.column_stack((features, mag_j_body_acc_entropy))
    mag_j_body_gyro_entropy = entropy_per_signal(mag_j_body_gyro)
    features = np.column_stack((features, mag_j_body_gyro_entropy))
    feature_names.append(['tGravityAcc-entropy()-X', 'tGravityAcc-entropy()-Y', 'tGravityAcc-entropy()-Z'])
    feature_names.append(['tBodyAcc-entropy()-X', 'tBodyAcc-entropy()-Y', 'tBodyAcc-entropy()-Z'])
    feature_names.append(['tBodyGyro-entropy()-X', 'tBodyGyro-entropy()-Y', 'tBodyGyro-entropy()-Z'])
    feature_names.append(['tBodyAccJerk-entropy()-X', 'tBodyAccJerk-entropy()-Y', 'tBodyAccJerk-entropy()-Z'])
    feature_names.append(['tBodyGyroJerk-entropy()-X', 'tBodyGyroJerk-entropy()-Y', 'tBodyGyroJerk-entropy()-Z'])
    feature_names.append(['tGravityAccMag-entropy()', 'tBodyAccMag-entropy()', 'tBodyGyroMag-entropy()', 'tBodyAccJerkMag-entropy()', 'tBodyGyroJerkMag-entropy()'])

    # AR COEFFICIENTS COMPUTING
    # calculate_arcoeff(grav_acc_x)
    grav_acc_x_arcoeff = calculate_arcoeff(grav_acc_x)
    for i in range(4):
        features = np.column_stack((features, grav_acc_x_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-X,' + str(i + 1))
    grav_acc_y_arcoeff = calculate_arcoeff(grav_acc_y)
    for i in range(4):
        features = np.column_stack((features, grav_acc_y_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-Y,' + str(i + 1))
    grav_acc_z_arcoeff = calculate_arcoeff(grav_acc_z)
    for i in range(4):
        features = np.column_stack((features, grav_acc_z_arcoeff[:, i]))
        feature_names.append('tGravityAcc-arCoeff()-Z,' + str(i + 1))
    body_acc_x_arcoeff = calculate_arcoeff(body_acc_x)
    for i in range(4):
        features = np.column_stack((features, body_acc_x_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-X,' + str(i + 1))
    body_acc_y_arcoeff = calculate_arcoeff(body_acc_y)
    for i in range(4):
        features = np.column_stack((features, body_acc_y_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-Y,' + str(i + 1))
    body_acc_z_arcoeff = calculate_arcoeff(body_acc_z)
    for i in range(4):
        features = np.column_stack((features, body_acc_z_arcoeff[:, i]))
        feature_names.append('tBodyAcc-arCoeff()-Z,' + str(i + 1))
    body_gyro_x_arcoeff = calculate_arcoeff(body_gyro_x)
    for i in range(4):
        features = np.column_stack((features, body_gyro_x_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-X,' + str(i + 1))
    body_gyro_y_arcoeff = calculate_arcoeff(body_gyro_y)
    for i in range(4):
        features = np.column_stack((features, body_gyro_y_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-Y,' + str(i + 1))
    body_gyro_z_arcoeff = calculate_arcoeff(body_gyro_z)
    for i in range(4):
        features = np.column_stack((features, body_gyro_z_arcoeff[:, i]))
        feature_names.append('tBodyGyro-arCoeff()-Z,' + str(i + 1))
    j_body_acc_x_arcoeff = calculate_arcoeff(j_body_acc_x)
    for i in range(4):
        features = np.column_stack((features, j_body_acc_x_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-X,' + str(i + 1))
    j_body_acc_y_arcoeff = calculate_arcoeff(j_body_acc_y)
    for i in range(4):
        features = np.column_stack((features, j_body_acc_y_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-Y,' + str(i + 1))
    j_body_acc_z_arcoeff = calculate_arcoeff(j_body_acc_z)
    for i in range(4):
        features = np.column_stack((features, j_body_acc_z_arcoeff[:, i]))
        feature_names.append('tBodyAccJerk-arCoeff()-Z,' + str(i + 1))
    j_body_gyro_x_arcoeff = calculate_arcoeff(j_body_gyro_x)
    for i in range(4):
        features = np.column_stack((features, j_body_gyro_x_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-X,' + str(i + 1))
    j_body_gyro_y_arcoeff = calculate_arcoeff(j_body_gyro_y)
    for i in range(4):
        features = np.column_stack((features, j_body_gyro_y_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-Y,' + str(i + 1))
    j_body_gyro_z_arcoeff = calculate_arcoeff(j_body_gyro_z)
    for i in range(4):
        features = np.column_stack((features, j_body_gyro_z_arcoeff[:, i]))
        feature_names.append('tBodyGyroJerk-arCoeff()-Z,' + str(i + 1))
    mag_grav_acc_arcoeff = calculate_arcoeff(mag_grav_acc)
    for i in range(4):
        features = np.column_stack((features, mag_grav_acc_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-X,' + str(i + 1))
    mag_body_acc_arcoeff = calculate_arcoeff(mag_body_acc)
    for i in range(4):
        features = np.column_stack((features, mag_body_acc_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-Y,' + str(i + 1))
    mag_body_gyro_arcoeff = calculate_arcoeff(mag_body_gyro)
    for i in range(4):
        features = np.column_stack((features, mag_body_gyro_arcoeff[:, i]))
        feature_names.append('tGravityAccMag-arCoeff()-Z,' + str(i + 1))
    mag_j_body_acc_arcoeff = calculate_arcoeff(mag_j_body_acc)
    for i in range(4):
        features = np.column_stack((features, mag_j_body_acc_arcoeff[:, i]))
        feature_names.append('tBodyAccMag-arCoeff()-X,' + str(i + 1))
    mag_j_body_gyro_arcoeff = calculate_arcoeff(mag_j_body_gyro)
    for i in range(4):
        features = np.column_stack((features, mag_j_body_gyro_arcoeff[:, i]))
        feature_names.append('tBodyGyroMag-arCoeff()-X,' + str(i + 1))

    # CORRELATION COMPUTING
    grav_acc_xy_corr = calculate_correlation(grav_acc_x, grav_acc_y)
    features = np.column_stack((features, grav_acc_xy_corr))
    grav_acc_xz_corr = calculate_correlation(grav_acc_x, grav_acc_z)
    features = np.column_stack((features, grav_acc_xz_corr))
    grav_acc_yz_corr = calculate_correlation(grav_acc_y, grav_acc_z)
    features = np.column_stack((features, grav_acc_yz_corr))
    body_acc_xy_corr = calculate_correlation(body_acc_x, body_acc_y)
    features = np.column_stack((features, body_acc_xy_corr))
    body_acc_xz_corr = calculate_correlation(body_acc_x, body_acc_z)
    features = np.column_stack((features, body_acc_xz_corr))
    body_acc_yz_corr = calculate_correlation(body_acc_y, body_acc_z)
    features = np.column_stack((features, body_acc_yz_corr))
    body_gyro_xy_corr = calculate_correlation(body_gyro_x, body_gyro_y)
    features = np.column_stack((features, body_gyro_xy_corr))
    body_gyro_xz_corr = calculate_correlation(body_gyro_x, body_gyro_z)
    features = np.column_stack((features, body_gyro_xz_corr))
    body_gyro_yz_corr = calculate_correlation(body_gyro_y, body_gyro_z)
    features = np.column_stack((features, body_gyro_yz_corr))
    j_body_acc_xy_corr = calculate_correlation(j_body_acc_x, j_body_acc_y)
    features = np.column_stack((features, j_body_acc_xy_corr))
    j_body_acc_xz_corr = calculate_correlation(j_body_acc_x, j_body_acc_z)
    features = np.column_stack((features, j_body_acc_xz_corr))
    j_body_acc_yz_corr = calculate_correlation(j_body_acc_y, j_body_acc_z)
    features = np.column_stack((features, j_body_acc_yz_corr))
    j_body_gyro_xy_corr = calculate_correlation(j_body_gyro_x, j_body_gyro_y)
    features = np.column_stack((features, j_body_gyro_xy_corr))
    j_body_gyro_xz_corr = calculate_correlation(j_body_gyro_x, j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_xz_corr))
    j_body_gyro_yz_corr = calculate_correlation(j_body_gyro_y, j_body_gyro_z)
    features = np.column_stack((features, j_body_gyro_yz_corr))
    feature_names.append(['tGravityAcc-correlation()-X,Y', 'tGravityAcc-correlation()-X,Z', 'tGravityAcc-correlation()-Y,Z'])
    feature_names.append(['tBodyAcc-correlation()-X,Y', 'tBodyAcc-correlation()-X,Z', 'tBodyAcc-correlation()-Y,Z'])
    feature_names.append(['tBodyGyro-correlation()-X,Y', 'tBodyGyro-correlation()-X,Z', 'tBodyGyro-correlation()-Y,Z'])
    feature_names.append(['tBodyAccJerk-correlation()-X,Y', 'tBodyAccJerk-correlation()-X,Z', 'tBodyAccJerk-correlation()-Y,Z'])
    feature_names.append(['tBodyGyroJerk-correlation()-X,Y', 'tBodyGyroJerk-correlation()-X,Z', 'tBodyGyroJerk-correlation()-Y,Z'])
    # print("n features: ", features.shape)
    feature_names_def = []
    for idx, val in enumerate(feature_names):
        if len(val) == 3 or len(val) == 5:
            for idx2, val2, in enumerate(val):
                feature_names_def.append(val2)
        else:
            feature_names_def.append(val)

    features_ordered = np.empty((features.shape[0],))
    to_overwrite = True
    feature_names_def_ordered = []
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAcc':
            if to_overwrite:
                features_ordered = features[:, idx]
                to_overwrite = False
            else:
                features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tGravityAcc':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccJerk':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyro':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroJerk':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tGravityAccMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyAccJerkMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    for idx, val in enumerate(feature_names_def):
        if val.split("-")[0] == 'tBodyGyroJerkMag':
            features_ordered = np.column_stack((features_ordered, features[:, idx]))
            feature_names_def_ordered.append(val)
    # features_ordered = np.array(features_ordered)
    # print(features_ordered)
    with open('/home/Scianso/tesi_server_flask/som_50.p', 'rb') as infile:
            mm_som = pickle.load(infile)

    def classify(som, data):
        """Classifies each sample in data in one of the classes definited
        using the method labels_map.
        Returns a list of the same length of data where the i-th element
        is the class assigned to data[i].
        """
        with open('/home/Scianso/tesi_server_flask/winmap.p', 'rb') as handle:
            winmap = pickle.load(handle)
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = som.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

    res = classify(mm_som, features_ordered)
    str_res = str(res[0])
    lbl_dict = {'0': 'walking', '1': 'walking upstairs', '2': 'walking downstairs', '3': 'sitting', '4': 'standing', '5': 'laying'}
    to_ret = lbl_dict[str_res]
    return to_ret