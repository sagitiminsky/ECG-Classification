from coding.dataset import Dataset
import ecg_plot
import numpy as np
import itertools
import pickle
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import copy
from google.cloud import storage
import os.path
import ast
import json
import uuid
import cv2
import pywt

## Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
client = storage.Client.from_service_account_json(json_credentials_path=path)

super_classes = {
    "CD": 0,
    "HYP": 1,
    "MI": 2,
    "NORM": 3,
    "STTC": 4

}


class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.DEBUG = False
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]
        self.state = 'HaarWavelet'  # expected {STFT,permutation,HaarWavelet}
        self.bucket = client.get_bucket('ecg-arrhythmia-classification')

        # datasttruct
        self.d = {
            "train": {
                0: {  # male
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }
                }
            },
            "test": {
                0: {  # male
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }
                }
            }
        }

        # Blend
        self.coeff_A = 0.5  # <-- how much does A effect the blending
        self.coeff_B = 0.5  # <-- how much does B effect the blending
        self.dataset_types = ["train", "test"]
        self.genders = [0, 1]
        self.ops = ["<", ">="]
        self.age_th = 50

        # STFT
        self.STFT_show = True
        self.hop = 1
        self.win = 128
        self.F = 1024
        self.sample_rate = 100

    def custom_operator(self, a, b, op):
        if op == "<":
            return a < b
        elif op == ">=":
            return a >= b

    def trancate(self, a, b):
        if len(a) != len(b):
            if len(a) > len(b):
                a = a[:len(b)]
            else:
                b = b[:len(a)]

        return a, b

    def feature_label_selection(self, d, gender, op, dataset_type):
        """
        extract the relevant features and lables both for train and predict
        """

        X = self.X_train if dataset_type == "train" else self.X_test
        X_meta = self.X_train_meta if dataset_type == "train" else self.X_test_meta
        Y = self.y_train if dataset_type == "train" else self.y_test

        mask_below = np.array(list(range(len(X)))) < len(X) // 2
        mask_above = list(reversed(mask_below))

        mask_A = np.isin(X_meta["sex"], gender) & self.custom_operator(
            X_meta["age"], self.age_th, op) & mask_below

        mask_B = np.isin(X_meta["sex"], gender) & self.custom_operator(
            X_meta["age"], self.age_th, op) & mask_above

        # features
        d[dataset_type][gender][op]["A"] = X[mask_A]
        d[dataset_type][gender][op]["meta_A"] = pd.DataFrame(X_meta)[mask_A].to_dict('records')

        d[dataset_type][gender][op]["B"] = X[mask_B]
        d[dataset_type][gender][op]["meta_B"] = pd.DataFrame(X_meta)[mask_B].to_dict('records')

        # labels
        a = Y[
            np.isin(X_meta["sex"], gender) & self.custom_operator(X_meta["age"],
                                                                  self.age_th,
                                                                  op) & mask_below]
        b = Y[
            np.isin(X_meta["sex"], gender) & self.custom_operator(X_meta["age"],
                                                                  self.age_th,
                                                                  op) & mask_above]

        return a, b

    def find_pairs(self):
        """
        pairs male with the same age(under/above 50)
        and female with the same age(under/above 50)
        :return:
        """

        d = copy.deepcopy(self.d)

        for dataset_type in self.dataset_types:
            for gender in self.genders:
                for op in self.ops:
                    a, b = self.feature_label_selection(d=d, gender=gender, op=op, dataset_type=dataset_type)

                    # trancate
                    a, b = self.trancate(a, b)

                    assert len(a)==len(b)

                    d[dataset_type][gender][op]["Y_A"], d[dataset_type][gender][op]["Y_B"] = a, b
                    d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"] = self.trancate(
                        d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"])
                    d[dataset_type][gender][op]["meta_A"], d[dataset_type][gender][op]["meta_B"] = self.trancate(
                        d[dataset_type][gender][op]["meta_A"],
                        d[dataset_type][gender][op]["meta_B"])

        return d

    def STFT(self, signal, win, hopSize, F, Fs):
        if not hasattr(win, "__len__"):
            win = np.hamming(win)
        if not hasattr(F, "__len__"):
            F = 2 * np.pi * np.arange(F) / F

        t = np.arange(len(signal))

        stft = []
        startIdx = 0
        while startIdx + len(win) <= len(signal):
            e = np.exp(
                -1j * t[startIdx:(startIdx + len(win))].reshape(1, -1) * F.reshape(-1, 1))
            currDFT = np.sum(signal[startIdx:(startIdx + len(win))] * win * e, 1)
            stft.append(np.abs(currDFT).astype(np.complex64))
            startIdx += hopSize

            if self.DEBUG and startIdx + len(win) > len(signal):
                print("iteration:True".format())

        stft = np.stack(stft).T
        return stft

    def haar_dwt(self,signal, levels=3):
        """
          Inputs:
            signal - input signal for analysis.
            levels – analysis depth.
          Outputs:
            approx - approximation of the signal at the last level.
            details - list with levels arrays. list[j] should contain the
            details of level j;
        """
        g_bar = 2 ** -0.5 * np.array([-1, 1])
        h = 2 ** -0.5 * np.array([1, 1])

        details = []
        approx = signal
        for ii in range(levels):
            details.append(np.convolve(approx, g_bar, 'valid')[::2])
            approx = np.convolve(approx, h, 'valid')[::2]

        return approx, details

    def gender_str(self, gender):
        return 'male' if gender == 0 else 'female'


    def standertize_and_normalize(self,mat):
        standard_mat = (mat - np.mean(mat)) / np.std(mat)
        max, min = np.max(standard_mat), np.min(standard_mat)
        return (standard_mat - min) / (max - min)

    def gcs_bucket(self, d):
        """
        According to the state, save a string that represent the data set.
        It is a string that can be later converterd to a dict. It contains the following groups {male,female},{<,>=}
        :param d: a dict containing the {male,female},{<,>=},{A,B,Y}
        :return: None
        """

        # Creating bucket object
        pkl_dict = copy.deepcopy(self.d)

        print("start gcs_bucket!")

        for dataset_type in self.dataset_types:
            for gender in self.genders:
                for op in self.ops:
                    assert len(d[dataset_type][gender][op]["A"]) == len(d[dataset_type][gender][op]["B"])
                    if self.state == "permutation":
                        for r, idx in zip(
                                itertools.product(d[dataset_type][gender][op]["A"],
                                                  d[dataset_type][gender][op]["B"]),
                                [a for a in itertools.product(*[range(len(x)) for x in
                                                                [d[dataset_type][gender][op]["A"],
                                                                 d[dataset_type][gender][op]["B"]]])]):

                            for idx_single, single in enumerate(["A", "B"]):
                                pkl_dict[dataset_type][gender][op][single].append(r[idx_single])
                                pkl_dict[dataset_type][gender][op][f"meta_{single}"].append(
                                    d[gender][op][f"meta_{single}"][idx[idx_single]])
                                pkl_dict[dataset_type][gender][op][f"Y_{single}"].append(
                                    d[gender][op][f"Y_{single}"][idx[idx_single]])
                    elif self.state == "STFT":
                        length = len(d[dataset_type][gender][op]["A"])
                        for index in range(length):
                            print("Now processing state:{} dataset_type:{}, gender:{}, op:{} img:{}/{} ".format(self.state,dataset_type, self.gender_str(gender), op,index + 1, length))

                            for single in ["A", "B"]:

                                ecg = d[dataset_type][gender][op][single][index].T[0]

                                X_stft = self.STFT(ecg, self.win, self.hop, self.F, self.sample_rate)

                                im = np.abs(X_stft)
                                middle_y = im.shape[1] // 2

                                #standertize and normalize
                                mat=self.standertize_and_normalize(im[middle_y:, 140:-145])

                                plt.imsave("myplot.jpeg", cv2.resize(mat,(256,256),interpolation=cv2.INTER_CUBIC))  #588,873

                                # Y
                                try:
                                    y = d[dataset_type][gender][op][f"Y_{single}"].iloc[index][0]
                                    pkl_dict[dataset_type][gender][op][f"Y_{single}"].append(super_classes[y])
                                except IndexError:
                                    print("Processing of state:{} index:{} failed due to Index error".format(self.state,index))
                                    break

                                # create the dataset: data+metadata
                                file_uuided = str(uuid.uuid4())

                                blob = self.bucket.blob('{}/{}.jpeg'.format(self.state, file_uuided))
                                with open("./myplot.jpeg", 'rb') as f:
                                    blob.upload_from_file(f)

                                pkl_dict[dataset_type][gender][op][single].append("{}/{}.jpeg".format(self.state,file_uuided))
                                pkl_dict[dataset_type][gender][op][f"meta_{single}"].append(
                                    d[dataset_type][gender][op][f"meta_{single}"][index])

                                if self.STFT_show:
                                    plt.show()

                            object_name_in_gcs_bucket = self.bucket.blob('data_map:{}'.format(self.state))
                            object_name_in_gcs_bucket.upload_from_string(str(pkl_dict))
                    elif self.state == "HaarWavelet":
                        length = len(d[dataset_type][gender][op]["A"])
                        for index in range(length):
                            print("Now processing state:{} dataset_type:{}, gender:{}, op:{} img:{}/{} ".format(self.state,dataset_type, self.gender_str(gender), op,index + 1, length))
                            for single in ["A", "B"]:

                                ecg = d[dataset_type][gender][op][single][index].T[0]
                                approximation,details= self.haar_dwt(ecg)
                                haar_dwt=np.concatenate([approximation,np.concatenate(details)])
                                mat=self.standertize_and_normalize(haar_dwt)

                                # Y
                                try:
                                    y = d[dataset_type][gender][op][f"Y_{single}"].iloc[index][0]
                                    pkl_dict[dataset_type][gender][op][f"Y_{single}"].append(super_classes[y])
                                except IndexError:
                                    print("Processing of state:{} index:{} failed due to Index error".format(self.state,
                                                                                                             index))
                                    break

                                file_uuided = str(uuid.uuid4())

                                object_name_in_gcs_bucket=self.bucket.blob('{}/{}'.format(self.state,file_uuided))
                                object_name_in_gcs_bucket.upload_from_string(str(mat))

                                pkl_dict[dataset_type][gender][op][single].append("{}/{}".format(self.state,file_uuided))
                                pkl_dict[dataset_type][gender][op][f"meta_{single}"].append(
                                    d[dataset_type][gender][op][f"meta_{single}"][index])

                            object_name_in_gcs_bucket = self.bucket.blob('data_map:{}'.format(self.state))
                            object_name_in_gcs_bucket.upload_from_string(str(pkl_dict))



                    else:
                        raise NotImplementedError



    def load_dataset(self):
        """
        https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object
        """

        blob = self.bucket.blob('data_map:{}'.format(self.state))
        d = ast.literal_eval(blob.download_as_string().decode('utf-8'))

        if self.state=='STFT':
            blob = self.bucket.blob("{}".format(d['train'][0]['<']['A'][0]))
            pickle.loads(blob.download_as_bytes())
            plt.show()
        elif self.state=='HaarWavelet':
            blob = self.bucket.blob("{}".format(d['train'][0]['<']['A'][0]))

            s=" ".join(blob.download_as_string().decode('utf-8').split())
            wavelet = ast.literal_eval(s.replace('\n','').replace(' ',','))
            print(wavelet)

    def blend_in_time(self, A, B):
        """
        Blend two matricies in time
        :param A: 12x1000 ndarray
        :param B: 12x1000 ndarray
        :return: C 12x1000 ndarray blended with self.coeff_A and self.coeff_B ratios
        """

        a = A[0]
        b = B[0]

        raise NotImplemented

    def blend_and_plot_ecg(self, pairs, index):
        """
        Display the ecg of a selected index
        :param pairs: a dict containing the {male,female},{<,>=},{A,B,Y} permutated
        :param index: the index of the pair for which we will preform the blending
        :return: None
        """

        gender = 0  # 0 male ; 1 female
        gender_str = "male" if not gender else "female"
        op = '<'
        op_str = "under 50" if op == '<' else "above 50"
        gender = pairs[gender]

        A = gender[op]["A"][index].T
        meta_A = gender[op]["meta_A"][index]
        B = gender[op]["B"][index].T
        meta_B = gender[op]["meta_B"][index]
        Y_A = gender[op]["Y_A"][index]
        Y_B = gender[op]["Y_B"][index]

        ecg_plot.plot(A, sample_rate=100, title="{}-{}-{}-{}".format(meta_A, gender_str, op_str, Y_A), columns=1)
        ecg_plot.plot(B, sample_rate=100, title="{}-{}-{}-{}".format(meta_B, gender_str, op_str, Y_B), columns=1)

        # <-- this is where the blending should happen

        # C=self.blend_in_time(A,B)

        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    pairs = b.find_pairs()
    b.gcs_bucket(pairs)
    b.load_dataset()
    # b.blend_and_plot_ecg(pairs, 0)
