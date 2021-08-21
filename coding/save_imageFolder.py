import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import scipy.signal as sg
import seaborn as sb
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm



def load_raw_data(df, sampling_rate, path):
    num_samples = 10
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr[:num_samples]]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr[:num_samples]]
    data = np.array([signal for signal, meta in data])
    return data


path = r'dataset\ptb-xl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sample_rate = 100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
# Y = Y[:num_samples]
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


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
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# X = load_raw_data(Y, sample_rate, path)

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

# data = np.array(wfdb.rdsamp(path + Y.filename_lr.iloc[0])[0])

hop = 1
win = 128
F = 1024
sample_rate = 100
# X_stft = STFT(data[200:800, 0], win, hop, F, sample_rate)
#
# tau = np.arange(X_stft.shape[1])*hop/sample_rate
# freqs = np.fft.fftshift(np.fft.fftfreq(F, 1/sample_rate))
# plt.figure()
# im = plt.pcolormesh(tau, freqs, np.fft.fftshift(np.abs(X_stft), axes=0), cmap='jet')
# plt.ylabel('f [Hz]', fontsize=16)
# plt.xlabel('$\\tau$ [sec]', fontsize=16)
# plt.title('win: ' + str(win) + '   hopSize: ' + str(hop) + '   F: ' + str(F), fontsize=16)
# plt.colorbar(im)
# plt.suptitle('| STFT(f, $\\tau$) |', fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])


save_path = './dataset/ptb-xl/images/'
for ii in tqdm(range(107, 4000)):
    data = np.array(wfdb.rdsamp(path + Y.filename_lr.iloc[ii])[0])
    stft = STFT(data[:, 0], win, hop, F, sample_rate)
    # cut to (512, 512)
    im = abs(stft[stft.shape[0] // 2:,
             ((stft.shape[1] - stft.shape[0] // 2) // 2):(-(stft.shape[1] - stft.shape[0] // 2) // 2)])
    # resize to (256, 256)
    im = cv2.resize(im, (256, 256))
    try:
        label = Y.diagnostic_superclass.iloc[ii][0]
        plt.imsave(save_path + label + '/' + str(ii) + '_age' + str(int(Y.age.iloc[ii])) + '_sex' + str(
            Y.sex.iloc[ii]) + '.jpeg', im)
        print('ii: ', ii)
    except:
        continue



