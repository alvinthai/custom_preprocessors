"""
OneHotLabelEncoder
"""

# Author: Alvin Thai <alvinthai@gmail.com>

from __future__ import division
from collections import defaultdict

import itertools
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class OneHotLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing a label encoding and one hot encoding sequentially.
    Can handle numerical, categorical, and missing inputs. Also recodes any
    unknown inputs found in transform step as an 'ignore_<feature_name>' value
    for one hot encoding. Fit and transform objects MUST be pandas DataFrame.

    Parameters
    ----------
    labels: list, optional
        Columns in DataFrame to fit OneHotLabelEncoder. If not provided,
        categorical columns from DataFrame will automatically be selected.

    top: int, float, or dict of numerics, optional
        Available option for reducing the cardinality of the LabelEncoder.
        - If int, selects <top> most frequent label values from each column.
        - If float (between 0 and 1), selects <top> percent most frequent label
          values from each column.
        - If dict (key = column label, value = int or float), user can
          explicitly set int or float <top> rules for specific columns.
        - If None (default), no label value filtering is performed.

        NOTE: <top> will not attempt to break ties. You may see more than <top>
              encoded labels if there is a tie in the value counts.

    concat: bool, optional, default: True
        Whether to concatenate the original data table to the DataFrame from
        the transform step of OneHotLabelEncoder.

    delete: boolean, optional, default: True
        Whether to delete the label column from DataFrame after transformation.

    ignore_col: boolean, optional, default: False
        Whether to output an indicator column for ignored values. Ignored
        values include unknown values found in the transform step and filtered
        values that do not satisfy the requirements for the top parameter.

    missing_col: boolean, optional, default: False
        Whether to output an indicator column for missing values, should
        specify to True if expecting missing values.
    '''
    def __init__(self, labels=None, top=None, concat=True, delete=True,
                 ignore_col=False, missing_col=False):
        self._keep = dict()
        self._le = defaultdict(LabelEncoder)

        self.labels = labels
        self.top = top
        self.concat = concat
        self.delete = delete
        self.ignore_col = ignore_col
        self.missing_col = missing_col

        if self.labels is None:
            self.labels = []

    def _fit_impute(self, X):
        '''
        Assigns and imputes ignored and missing values for each feature that
        LabelEncoder will be fitting.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.
        '''
        n = len(X)

        self._impute_ignore = dict()
        self._impute_missing = dict()
        self._n_unq = dict()

        for col in self.labels:
            if X[col].dtype == object:
                self._impute_ignore[col] = u'\uffff'
                self._impute_missing[col] = u'\uffff'u'\uffff'
            else:
                col_max = X[col].max()
                self._impute_ignore[col] = col_max + 1
                self._impute_missing[col] = col_max + 2

            missing = X[col].isnull()

            if col in self._keep:  # ignore unknown non-missing values
                ignore = np.logical_and(~X[col].isin(self._keep[col]),
                                        ~missing)
                self._n_unq[col] = len(self._keep[col])
            else:  # ignore nothing
                ignore = np.zeros(n).astype(bool)
                self._n_unq[col] = len(X[col].unique()) - (missing.sum() > 0)

            X.loc[ignore, col] = self._impute_ignore[col]
            X.loc[missing, col] = self._impute_missing[col]

    def _fit_reduce_cardinality(self, X, cols, top):
        '''
        Reduces cardinality of the LabelEncoder by selecting the <top> most
        common label values or <top> percent most frequent label values from
        encoded columns. Values not belonging to the <top> label values will
        be treated as unknown values to OneHotLabelEncoder and are assigned to
        the 'ignore_<feature_name>' column.

        NOTE: <top> will not attempt to break ties. You may see more than <top>
              encoded labels if there is a tie in the value counts.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.

        cols: list
            Columns in DataFrame to fit OneHotLabelEncoder.

        top: int, float, or dict of numerics, optional
            Available option for reducing the cardinality of the LabelEncoder.
            - If int, selects <top> most frequent label values from each column
            - If float (between 0 and 1), selects <top> percent most frequent
              label values from each column.
            - If dict (key = column label, value = int or float), user can
              explicitly set int or float <top> rules for specific columns.
            - If None (default), no label value filtering is performed.
        '''
        mode = 'none'

        if isinstance(top, (int, float, type(None))):
            if top >= 1:
                mode = 'int'
            elif top > 0:
                mode = 'ratio'
        elif isinstance(top, dict):
            # recursive call
            for k, v in top.iteritems():
                self._fit_reduce_cardinality(X, [k], v)

        if mode == 'int':
            for col in cols:
                y = X[col]

                if top >= len(y.unique()):
                    continue

                # get descending value counts
                cnt_desc = y.value_counts()

                # keep top n distinct values and ties
                keeps = cnt_desc >= cnt_desc.iloc[top-1]

                self._keep[col] = set(keeps[keeps].index)

        elif mode == 'ratio':
            for col in cols:
                y = X[col]

                # get descending value counts and ratios
                cnt_desc, ratios = self._get_value_count_desc(y)

                # keep distinct values that make up top % of data and ties
                keep = ratios < top
                keep = np.logical_or(keep,
                                     cnt_desc == cnt_desc.iloc[sum(keep) - 1])

                self._keep[col] = set(keep[keep].index)

    def _get_categorical(self, X):
        '''
        Executes when no labels are specified by user during initation.
        Automatically chooses columns to perform OneHotLabelEncoder based on
        the data type of column values.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.
        '''
        n = len(X)

        for col, dtype in X.dtypes.iteritems():
            if dtype == object:
                # check if all values in column are distinct
                if len(X[col].unique()) == n:
                    continue
                else:
                    self.labels.append(col)

    def _get_value_count_desc(self, y):
        '''
        Gets histogram values (in descending order) and percentage of data
        belonging to more frequent label values for a categorical column.

        Parameters
        ----------
        y: Series, shape = [n_samples, ]
            Input column to calculate histogram values for.

        Returns
        -------
        cnt_desc: Series, shape = [n_unique_values, ]
            Histogram values (in descending order) of unique labels from column

        ratios: Series, shape = [n_unique_values, ]
            Tells us the start index if the data column was sorted in
            descending order by the histogram count. This number is normalized
            into [0, 1) range to output percentage of data belonging to label
            values occurring more frequently than the current label.
        '''
        cnt = len(y)
        cnt_desc = y.value_counts()

        cnt_desc_hist_idx = cnt_desc.shift(1).cumsum().fillna(0)
        ratios = cnt_desc_hist_idx / cnt

        return cnt_desc, ratios

    def _name_columns(self, X_out):
        '''
        Function for naming columns in OneHotEncoder DataFrame.
        - Prepends feature name to label values from OneHotLabelEncoder.
        - If value is ignored, prepends ignore to feature name.
        - If value is missing, prepends missing to feature name.

        Parameters
        ----------
        X_out: DataFrame, shape = [n_samples, encoded_features]
            Transfomed DataFrame from OneHotEncoder with unnamed columns.

        Returns
        -------
        X_out: DataFrame, shape = [n_samples, encoded_features]
            Transformed DataFrame from OneHotEncoder with named columns.
        '''
        cols = np.repeat(self.labels, self._ohe.n_values_).astype(object)

        indexes = zip(self._ohe.feature_indices_,
                      self._ohe.feature_indices_[1:] - 2)

        for i, (j, k) in enumerate(indexes):
            lbl = self.labels[i]
            vals = self._le[lbl].classes_[:k-j]

            subcols = np.char.array(cols[j:k]) + '_' + vals.astype(str)

            cols[j:k] = subcols
            cols[k] = 'ignore_' + str(lbl)
            cols[k+1] = 'missing_' + str(lbl)

        X_out.columns = cols
        return X_out

    def _select_columns(self, X, X_out):
        '''
        Selects columns to output when returning transform results.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Original input DataFrame from transform step.

        X_out: DataFrame, shape = [n_samples, encoded_features]
            Transformed DataFrame from OneHotEncoder with named columns.

        Returns
        -------
        X_out: DataFrame, shape = [n_samples, (n_features +) encoded_features]
            Original DataFrame with OneHotLabelEncoder columns.
        '''
        cols_out = np.arange(X_out.shape[1])
        ohe_idx = self._ohe.feature_indices_
        ohe_classes = self._ohe.n_values_

        if not self.ignore_col:
            drop = ohe_idx[:-1] + ohe_classes - 2
            cols_out = np.setdiff1d(cols_out, drop)

        if not self.missing_col:
            drop = ohe_idx[:-1] + ohe_classes - 1
            cols_out = np.setdiff1d(cols_out, drop)

        if not self.concat:
            return X_out.iloc[:, cols_out]
        else:
            X_out = pd.concat([X, X_out], axis=1)

        n_cols_in = X.shape[1]
        cols_in = np.arange(n_cols_in).tolist()

        # reorder columns so that encoded columns appear in-place
        label_idx = [X.columns.tolist().index(lbl) for lbl in self.labels]
        zipped = sorted(zip(label_idx, ohe_idx, ohe_idx[1:]), reverse=True)
        cols_in = map(lambda c: [c], cols_in)

        for i, j, k in zipped:
            mask = np.logical_and(cols_out >= j, cols_out < k)
            cols_in.insert(i + 1, cols_out[mask] + n_cols_in)

        cols_in = list(itertools.chain(*cols_in))
        X_out = X_out.iloc[:, cols_in]

        if self.delete:
            X_out = X_out.drop(self.labels, axis=1)

        return X_out

    def _transform_impute(self, X):
        '''
        Imputes ignored and missing values prior to LabelEncoder transform.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.
        '''
        n = len(X)

        for col in self.labels:
            classes = self._le[col].classes_
            n_unq = self._n_unq[col]

            missing = X[col].isnull()
            ignore = ~X[col].isin(set(classes[:n_unq]))
            ignore = np.logical_or(ignore, X[col].isin(set(classes[n_unq:])))

            X.loc[ignore, col] = self._impute_ignore[col]
            X.loc[missing, col] = self._impute_missing[col]

    def fit(self, X, y=None):
        '''
        Fits OneHotLabelEncoder to X.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.

        y: optional
            Passthrough for Pipeline compatibility.

        Returns
        -------
        self
        '''
        X = X.copy()
        n_values = []

        if len(self.labels) == 0:
            self._get_categorical(X)

        self._fit_reduce_cardinality(X, self.labels, self.top)
        self._fit_impute(X)

        for col in self.labels:
            n_unq = self._n_unq[col]
            v_ignore = self._impute_ignore[col]
            v_miss = self._impute_missing[col]

            # specify number of classes to expect for OneHotEncoder
            n_values.append(n_unq + 2)

            enc = self._le[col]
            enc.fit(X[col])  # fit LabelEncoder for column

            # overwrite classes_ attribute with imputation labels
            enc.classes_ = np.hstack([enc.classes_[:n_unq], v_ignore, v_miss])
            # transform LabelEncoder prior to fitting OneHotEncoding
            X[col] = enc.transform(X[col])

        if len(self.labels) == 1:
            X = X[self.labels].values.reshape(-1, 1)
        else:
            X = X[self.labels]

        self._ohe = OneHotEncoder(n_values=n_values, sparse=False)
        self._ohe.fit(X)

        return self

    def transform(self, X):
        '''
        Performs OneHotLabelEncoder transform operation on the input DataFrame.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.

        Returns
        -------
        X_out: DataFrame, shape = [n_samples, (n_features +) encoded_features]
            Original DataFrame with OneHotLabelEncoder columns.
        '''
        X_out = X.copy()
        self._transform_impute(X_out)

        for col in self.labels:
            X_out[col] = self._le[col].transform(X_out[col])

        if len(self.labels) == 1:
            X_out = X_out[self.labels].values.reshape(-1, 1)
        else:
            X_out = X_out[self.labels]

        X_out = pd.DataFrame(self._ohe.transform(X_out), index=X.index)
        X_out = self._name_columns(X_out).astype(int)
        X_out = self._select_columns(X, X_out)

        return X_out
