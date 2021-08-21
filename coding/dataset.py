import pandas as pd
import numpy as np
import wfdb
import ast


class Dataset:
    def __init__(self):

        self.path = './toy_data/'
        self.sampling_rate = 100

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.path + 'scp_statements.csv', index_col=0)
        self.agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def load_raw_data(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def extract_meta(self, Y, test_fold, case):

        if case == "Train":
            return {
                "age": Y[~np.isin(Y.strat_fold, test_fold)].age,
                "sex": Y[~np.isin(Y.strat_fold, test_fold)].sex,
                "patient_id": Y[~np.isin(Y.strat_fold, test_fold)].patient_id
            }
        else:
            return {
                "age": Y[np.isin(Y.strat_fold, test_fold)].age,
                "sex": Y[np.isin(Y.strat_fold, test_fold)].sex,
                "patient_id": Y[np.isin(Y.strat_fold, test_fold)].patient_id
            }

    def load(self):
        # load and convert annotation data
        Y = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = self.load_raw_data(Y, self.sampling_rate, self.path)

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        # Split data into train and test
        test_fold = [10]
        # Train
        X_train = X[~np.isin(Y.strat_fold, test_fold)]
        X_train_meta = self.extract_meta(Y, test_fold, case="Train")
        y_train = Y[~np.isin(Y.strat_fold, test_fold)].diagnostic_superclass

        # Test
        X_test = X[np.isin(Y.strat_fold, test_fold)]
        X_test_meta = self.extract_meta(Y, test_fold, case="Test")
        y_test = Y[np.isin(Y.strat_fold, test_fold)].diagnostic_superclass

        return {
            "X_train": X_train,
            "X_train_meta": X_train_meta,
            "y_train": y_train,
            "X_test": X_test,
            "X_test_meta": X_test_meta,
            "y_test": y_test
        }
