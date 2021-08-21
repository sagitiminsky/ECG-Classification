import io
import PIL.Image as Image
import ast
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytorch_lightning as pl

from google.cloud import storage
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import random


# Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
# client = storage.Client.from_service_account_json(json_credentials_path=path)
client = storage.Client.from_service_account_json(json_credentials_path='model-azimuth-321409-241148a4b144.json')
bucket = client.get_bucket('ecg-arrhythmia-classification')


class Pattern():
    def __init__(self, enc, percent):
        self.enc = enc
        self.percent = percent


class PatternSelection():
    def __init__(self):
        """
        This class enables you to select a percentage of the data from each class

        """
        self.patterns = {
            "CD": Pattern(enc=0, percent=1),
            "HYP": Pattern(enc=1, percent=1),
            "MI": Pattern(enc=2, percent=1),
            "NORM": Pattern(enc=3, percent=0.33),
            "STTC": Pattern(enc=4, percent=1)
        }
        assert all([self.percent_is_valid(pattern.percent) for pattern in self.patterns.values()])

    def percent_is_valid(self, percent):
        if percent > 0 and percent <= 1:
            return True
        return False


class PtbData(Dataset):

    def __init__(self, data_map_url, data_url, gender, under_50, is_train, download=False, transform=None):
        """
        Args:

        """

        self.data_url = data_url
        self.gender = gender
        self.under_50 = under_50
        self.transform = transform
        self.state = 'train' if is_train else 'test'
        self.download = download
        self.root_images = './STFT'
        self.feature_selection = PatternSelection()

        if self.download:
            blob_obj = bucket.blob(data_map_url)
            self.data_map = ast.literal_eval(blob_obj.download_as_string().decode('utf-8'))

            self.features, self.labels, self.metadata = self.feature_label_metadata_selection()

            with open('data_map.pickle', 'wb') as handle:
                pickle.dump(self.data_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('labels.pickle', 'wb') as handle:
                pickle.dump(self.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if not os.path.exists(self.root_images):
                os.makedirs(self.root_images)
                length = len(self.features)

                for ii in range(length):
                    blob_obj = bucket.blob("{}".format(self.features[ii]))
                    sample = Image.open(io.BytesIO(blob_obj.download_as_bytes())).convert('L')
                    age = int(self.metadata[ii]['age'])
                    sex = self.metadata[ii]['sex']

                    if ii % 100 == 0:
                        print("Saving image:{}/{}".format(ii, length))
                    sample.save('{}'.format(self.features[ii]))



        else:
            with open('data_map.pickle', 'rb') as handle:
                self.data_map = pickle.load(handle)

            with open('labels.pickle', 'rb') as handle:
                self.labels = pickle.load(handle)

        self.files = os.listdir(self.root_images)

    def feature_label_metadata_selection(self):
        features, labels, metadata = [], [], []
        for pattern in self.feature_selection.patterns.values():
            percent = pattern.percent
            enc = pattern.enc
            mask_A = np.array(self.data_map[self.state][self.gender][self.under_50]['Y_A']) == enc
            mask_B = np.array(self.data_map[self.state][self.gender][self.under_50]['Y_B']) == enc

            features += self.select_from_mask_and_merge(self.data_map[self.state][self.gender][self.under_50]['A'],
                                                        self.data_map[self.state][self.gender][self.under_50]['B'],
                                                        mask_A, mask_B, percent)
            labels += self.select_from_mask_and_merge(self.data_map[self.state][self.gender][self.under_50]['Y_A'],
                                                      self.data_map[self.state][self.gender][self.under_50]['Y_B'],
                                                      mask_A, mask_B, percent)
            metadata += self.select_from_mask_and_merge(self.data_map[self.state][self.gender][self.under_50]['meta_A'],
                                                        self.data_map[self.state][self.gender][self.under_50]['meta_B'],
                                                        mask_A, mask_B, percent)

        c = list(zip(features, labels, metadata))
        for i in range(5):
            random.shuffle(c)
        features, labels, metadata = zip(*c)

        return features, labels, metadata

    def select_from_mask_and_merge(self, list_A, list_B, mask_A, mask_B, percent):
        length_A = len(list_A)
        length_B = len(list_B)
        list_A = np.array(list_A)
        list_B = np.array(list_B)

        return (list(list_A[mask_A]) + list(list_B[mask_B]))[:int(percent * (length_A + length_B))]

    def __len__(self):
        # len(self.data_map[self.state][self.gender][self.under_50]['A']) # +\
        # len(self.data_map[self.state][self.gender][self.under_50]['B'])
        return len(self.files)

    def __getitem__(self, idx):
        sample = Image.open('./STFT/' + self.files[idx])
        y = int(self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, y


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_map_url, data_url, gender, under_50, is_train, transform=None):
        super().__init__()
        self.batch_size = batch_size

        self.data_map_url = data_map_url
        self.data_url = data_url
        self.gender = 0 if gender == "male" else 1
        self.under_50 = '<' if under_50 else '>='
        self.state = 'train' if is_train else 'test'

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        # download
        PtbData(self.data_map_url, self.data_url, self.gender, self.under_50, self.state, download=True,
                transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'train' or stage is None:
            ptb_full = PtbData(self.data_map_url, self.data_url, self.gender, self.under_50, 'train',
                               transform=self.transform)
            self.train, self.val = random_split(ptb_full, [round(len(ptb_full) * 0.8),
                                                           len(ptb_full) - round(len(ptb_full) * 0.8)])
            print('len ptb_full: ', len(ptb_full))
            print('len train: ', len(self.train))
            print('len val: ', len(self.val))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = PtbData(self.data_map_url, self.data_url, self.gender, self.under_50, 'test',
                                transform=self.transform)
            print('len test: ', len(self.test))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        # return DataLoader(self.test, batch_size=self.batch_size)
        return DataLoader(self.val, batch_size=self.batch_size)



