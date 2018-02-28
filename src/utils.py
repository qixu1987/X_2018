from sklearn.base import BaseEstimator, TransformerMixin
import sys


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
