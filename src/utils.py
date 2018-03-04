from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import numpy as np


class WarrantyToFloat(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y = 0):
        return self

    def transform(self, input_pds, y = 0):
        pds = input_pds.copy()
        pds = pds.str[0].astype(int)
        return pds


class FillByMax(TransformerMixin, BaseEstimator):
    def __init__(self, max_float=(2**63-1)):
        self.max_float = max_float

    def fit(self, X, y = 0):
        return self

    def transform(self, input_df, y = 0):
        df = input_df.copy()
        df = df.fillna(self.max_float)
        return df


def one_folder_out(df_train_prepro, target, regressor, cv_number, random_seed):
    prediction = np.array(np.repeat(np.nan, df_train_prepro.shape[0]))
    kf = KFold(n_splits=cv_number, shuffle=False, random_state=random_seed)
    for train_index, test_index in kf.split(df_train_prepro):
        x_train, x_test = df_train_prepro.iloc[train_index, :], df_train_prepro.iloc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        clf_k = regressor
        clf_k.fit(x_train, y_train)
        prediction[test_index] = clf_k.predict(x_test)
    return prediction
