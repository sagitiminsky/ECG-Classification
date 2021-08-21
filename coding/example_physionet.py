import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import scipy.signal as sg
import seaborn as sb
import pandas as pd
num_samples = 10



def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr[:num_samples]]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr[:num_samples]]
    data = np.array([signal for signal, meta in data])
    return data


mit_bih_signal = pd.read_csv('dataset/mit-bih/N/100.csv')
sig = mit_bih_signal.values[1000:(1000+10*360), 1]

path = r'dataset\ptb-xl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sample_rate = 100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y = Y[:num_samples]
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sample_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes[:num_samples].apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[Y.strat_fold != test_fold].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass


def STFT(signal, win, hopSize, F, Fs):
    if not hasattr(win, "__len__"):
        win = np.hamming(win)
    if not hasattr(F, "__len__"):
        F = 2*np.pi*np.arange(F)/F

    t = np.arange(len(signal))

    stft = []
    startIdx = 0
    while startIdx + len(win) <= len(signal):
        e = np.exp(-1j * t[startIdx:(startIdx + len(win))].reshape(1, -1) * F.reshape(-1, 1))
        currDFT = np.sum(signal[startIdx:(startIdx + len(win))]*win*e, 1)
        stft.append(np.abs(currDFT).astype(np.complex64))
        startIdx += hopSize

    stft = np.stack(stft).T
    return stft
#
# f, t, Zxx = sg.stft(X_train[7, :, 0], fs=100, nperseg=512, noverlap=512-1)
# f, t, Zxx = sg.stft(rec["'MLII'"][:1024], fs=360, nperseg=512, noverlap=0)
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')

# hop = 1
# win = 64
# F = 512
# resample_rate = 100
# X_stft = STFT(X_train[0, :, 0], win, hop, F, sample_rate)


hop = 1
win = 128
F = 1024
sample_rate = 360
X_stft = STFT(sig-min(sig), win, hop, F, sample_rate)

tau = np.arange(X_stft.shape[1])*hop/sample_rate
freqs = np.fft.fftshift(np.fft.fftfreq(F, 1/sample_rate))
im = plt.pcolormesh(tau, freqs, np.fft.fftshift(np.abs(X_stft), axes=0), cmap='jet')
plt.ylabel('f [Hz]', fontsize=16)
plt.xlabel('$\\tau$ [sec]', fontsize=16)
plt.title('win: ' + str(win) + '   hopSize: ' + str(hop) + '   F: ' + str(F), fontsize=16)
plt.colorbar(im)
plt.suptitle('| STFT(f, $\\tau$) |', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




for ii in range(X_train.shape[0]):
    plt.figure(figsize=(20,3))
    # plt.plot(np.fft.fftshift(mag))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(X_train[ii, :, 0]), 1/sample_rate))
    plt.plot(freqs, np.abs(np.fft.fftshift(np.fft.fft(X_train[ii, :, 0]))))
    # plt.xticks(np.arange(-2000, 2000, 500), np.arange(-2000, 2000, 500))
    plt.title("Frequency Domain - " + y_train.iloc[ii][0], fontsize=16)
    plt.xlabel("Frequency [Hz]", fontsize=16)
    # plt.ylabel("|Fourier Coefficient|")
    plt.ylabel("$X^f(f)$", fontsize=16)
    plt.grid()
    # plt.show()



