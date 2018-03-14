"""
Custom Preprocessors for:
    - CountEncoder
    - MultinomialNBEncoder
    - TargetEncoder
    - OneHotLabelEncoder
"""

# Author: Alvin Thai <alvinthai@gmail.com>

from __future__ import division
from collections import defaultdict

import itertools
import numpy as np
import pandas as pd
import re

from encoder_utils import GridMultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CountEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing count encoding on categorical variables.
    If a new category group is found during transform, -1 is returned.

    Fit and transform objects MUST be pandas DataFrame.

    Parameters
    ----------
    cols: list
        Columns in DataFrame to fit MultinomialNBEncoder.
    '''
    def __init__(self, cols):
        self.encoder_dict = {}
        self.cols = cols

    def fit(self, df, y=None):
        '''
        Fits CountEncoder to df.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting CountEncoder.

        y: optional
            Passthrough for Pipeline compatibility.

        Returns
        -------
        self
        '''
        for col in self.cols:
            self.encoder_dict[col] = df[col].value_counts().to_dict()

        return self

    def transform(self, df):
        '''
        Performs transform operation on the input DataFrame.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.

        Returns
        -------
        df: DataFrame, shape = [n_samples, n_features + encoded_features]
            Original DataFrame with CountEncoder columns.
        '''
        df = df.copy()

        for col in self.cols:
            func = lambda x: self.encoder_dict[col].get(x, -1)
            df['COUNT(' + col + ')'] = df[col].map(func)

        return df


class MultinomialNBEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing Multinomial Naive Bayes classification on text data.
    Fit and transform objects MUST be pandas DataFrame.

    Parameters
    ----------
    cols: list
        Columns in DataFrame to fit MultinomialNBEncoder.
    '''
    def __init__(self, cols):
        self.encoder_dict = {}
        self.cols = cols

    def fit(self, df, y):
        '''
        Fits MultinomialNBEncoder to df.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting MultinomialNBEncoder.

        y: array-like, shape = [n_samples, ]
            True labels for df.

        Returns
        -------
        self
        '''
        for col in self.cols:
            self.encoder_dict[col] = GridMultinomialNB().fit(df[col], y)

        return self

    def transform(self, df):
        '''
        Performs transform operation on the input DataFrame.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.

        Returns
        -------
        df: DataFrame, shape = [n_samples, n_features + encoded_features]
            Original DataFrame with MultinomialNBEncoder columns.
        '''
        df = df.copy()

        for col in self.cols:
            model = self.encoder_dict[col]
            classes = len(model.model.class_count_)

            # use proba for positive class when binary, otherwise use predict
            if classes == 2:
                mnb_output = model.predict_proba(df[col])[:, 1]
            else:
                mnb_output = model.predict(df[col])

            df['MultinomialNB(' + col + ')'] = mnb_output

        return df


class TargetEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing target encoding on categorical variables.
    By default, TargetEncoder returns the target mean for the category group.
    If a new category group is found during transform, the specified <unknown>
    value is returned.

    Optional parameters include different aggregation functions (mean, median,
    min, max, std, sum, or quantile), and a <smoothing_size> parameter to
    return a weighted mean of AGG(y) when group sample sizes are insufficient.

    Fit and transform objects MUST be pandas DataFrame.

    Parameters
    ----------
    cols: list
        Columns in DataFrame to fit TargetEncoder.

    smoothing_size: int, optional, default: 0
        Sample size for calculating weighted mean on aggregated statistic.
        - If COUNT(group) >= smoothing_size, no weighted means are calculated.
        - If COUNT(group) < smoothing_size, a weighted mean is calculated with AGG(y):
          i.e. with smoothing_size=100, COUNT(group)=10, AGG(group)=0.8, AGG(y)=0.1
               output returns --> (10*0.8 + (100-10)*0.1) / 100 = 0.17

    agg: str, optional, default: 'mean'
        Aggregation options for target encoding. Available options:
        - mean: 'mean'
        - median: 'median'
        - minimum: 'min'
        - maximum: 'max'
        - standard deviation: 'std'
        - sum: 'sum'
        - quantile (input desired quantile in ##): 'q##'

    unknown: numeric, optional, default: -1
        Value to return when unknown category group found during transform.
    '''
    def __init__(self, cols, smoothing_size=0, agg='mean', unknown=-1):
        self.encoder_dict = {}
        self.cols = cols
        self.smoothing_size = smoothing_size
        self.agg = agg.title()
        self.unknown = unknown

        if agg == 'mean':
            self.aggfunc = lambda x: x.mean()
        elif agg == 'median':
            self.aggfunc = lambda x: x.median()
        elif agg == 'min':
            self.aggfunc = lambda x: x.min()
        elif agg == 'max':
            self.aggfunc = lambda x: x.max()
        elif agg == 'std':
            self.aggfunc = lambda x: x.std()
        elif agg == 'sum':
            self.aggfunc = lambda x: x.sum()
        elif re.match('q[0-9]{1,2}$', agg) is not None:
            q = int(agg[1:])
            self.aggfunc = lambda x: x.quantile(q/100)
        else:
            raise ValueError('invalid value for {agg}')

    def fit(self, df, y):
        '''
        Fits TargetEncoder to df.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting TargetEncoder.

        y: array-like, shape = [n_samples, ]
            True labels for df.

        Returns
        -------
        self
        '''
        overall_wgt = lambda x: max(self.smoothing_size-x, 0)
        smoothed_sample_size = lambda x: max(self.smoothing_size, x)

        for col in self.cols:
            x = pd.DataFrame()
            x[col] = df[col]
            x['target'] = pd.Series(y)

            g = x.groupby(col)
            yagg = self.aggfunc(y)

            g_df = pd.DataFrame(index=sorted(g.groups.keys()))
            g_df['agg'] = self.aggfunc(g)
            g_df['count'] = g.count()

            # Calculates contribution to means with respect to mix size
            # if COUNT(group) >= smoothing_size:
            #   AGG(group) is returned
            # if COUNT(group) < smoothing_size:
            #   a weighted average of AGG(group) and AGG(y) is returned
            #   the influence of AGG(y) increases with lower COUNT(group)
            tgt_contrib = g_df['count'] * g_df['agg']
            overall_contrib = g_df['count'].map(overall_wgt) * yagg

            numerator = tgt_contrib + overall_contrib
            denominator = g_df['count'].map(smoothed_sample_size)
            g_df['smoothed_mean'] = numerator / denominator

            self.encoder_dict[col] = g_df['smoothed_mean'].to_dict()

        return self

    def transform(self, df):
        '''
        Performs transform operation on the input DataFrame.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.

        Returns
        -------
        df: DataFrame, shape = [n_samples, n_features + encoded_features]
            Original DataFrame with TargetEncoder columns.
        '''
        df = df.copy()

        for col in self.cols:
            label = 'Target{}({})'.format(self.agg, col)
            func = lambda x: self.encoder_dict[col].get(x, self.unknown)
            df[label] = df[col].map(func)

        return df


class OneHotLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing a label encoding and one hot encoding sequentially.
    Can handle numerical, categorical, and missing inputs. Also recodes any
    unknown inputs found in transform step as an 'ignore_<feature_name>' value
    for one hot encoding.

    Fit and transform objects MUST be pandas DataFrame.

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

    def _fit_impute(self, df):
        '''
        Assigns and imputes ignored and missing values for each feature that
        LabelEncoder will be fitting.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.
        '''
        n = len(df)

        self._impute_ignore = dict()
        self._impute_missing = dict()
        self._n_unq = dict()

        for col in self.labels:
            if df[col].dtype == object:
                self._impute_ignore[col] = u'\uffff'
                self._impute_missing[col] = u'\uffff'u'\uffff'
            else:
                col_max = df[col].max()
                self._impute_ignore[col] = col_max + 1
                self._impute_missing[col] = col_max + 2

            missing = df[col].isnull()

            if col in self._keep:  # ignore unknown non-missing values
                ignore = np.logical_and(~df[col].isin(self._keep[col]),
                                        ~missing)
                self._n_unq[col] = len(self._keep[col])
            else:  # ignore nothing
                ignore = np.zeros(n).astype(bool)
                self._n_unq[col] = len(df[col].unique()) - (missing.sum() > 0)

            df.loc[ignore, col] = self._impute_ignore[col]
            df.loc[missing, col] = self._impute_missing[col]

    def _fit_reduce_cardinality(self, df, cols, top):
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
        df: DataFrame, shape = [n_samples, n_features]
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
                self._fit_reduce_cardinality(df, [k], v)

        if mode == 'int':
            for col in cols:
                y = df[col]

                if top >= len(y.unique()):
                    continue

                # get descending value counts
                cnt_desc = y.value_counts()

                # keep top n distinct values and ties
                keeps = cnt_desc >= cnt_desc.iloc[top-1]

                self._keep[col] = set(keeps[keeps].index)

        elif mode == 'ratio':
            for col in cols:
                y = df[col]

                # get descending value counts and ratios
                cnt_desc, ratios = self._get_value_count_desc(y)

                # keep distinct values that make up top % of data and ties
                keep = ratios < top
                keep = np.logical_or(keep,
                                     cnt_desc == cnt_desc.iloc[sum(keep) - 1])

                self._keep[col] = set(keep[keep].index)

    def _get_categorical(self, df):
        '''
        Executes when no labels are specified by user during initation.
        Automatically chooses columns to perform OneHotLabelEncoder based on
        the data type of column values.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.
        '''
        n = len(df)

        for col, dtype in df.dtypes.iteritems():
            if dtype == object:
                # check if all values in column are distinct
                if len(df[col].unique()) == n:
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

    def _name_columns(self, df_out):
        '''
        Function for naming columns in OneHotEncoder DataFrame.
        - Prepends feature name to label values from OneHotLabelEncoder.
        - If value is ignored, prepends ignore to feature name.
        - If value is missing, prepends missing to feature name.

        Parameters
        ----------
        df_out: DataFrame, shape = [n_samples, encoded_features]
            Transfomed DataFrame from OneHotEncoder with unnamed columns.

        Returns
        -------
        df_out: DataFrame, shape = [n_samples, encoded_features]
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

        df_out.columns = cols
        return df_out

    def _select_columns(self, df, df_out):
        '''
        Selects columns to output when returning transform results.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Original input DataFrame from transform step.

        df_out: DataFrame, shape = [n_samples, encoded_features]
            Transformed DataFrame from OneHotEncoder with named columns.

        Returns
        -------
        df_out: DataFrame, shape = [n_samples, (n_features +) encoded_features]
            Original DataFrame with OneHotLabelEncoder columns.
        '''
        cols_out = np.arange(df_out.shape[1])
        ohe_idx = self._ohe.feature_indices_
        ohe_classes = self._ohe.n_values_

        if not self.ignore_col:
            drop = ohe_idx[:-1] + ohe_classes - 2
            cols_out = np.setdiff1d(cols_out, drop)

        if not self.missing_col:
            drop = ohe_idx[:-1] + ohe_classes - 1
            cols_out = np.setdiff1d(cols_out, drop)

        if not self.concat:
            return df_out.iloc[:, cols_out]
        else:
            df_out = pd.concat([df, df_out], axis=1)

        n_cols_in = df.shape[1]
        cols_in = np.arange(n_cols_in).tolist()

        # reorder columns so that encoded columns appear in-place
        label_idx = [df.columns.tolist().index(lbl) for lbl in self.labels]
        zipped = sorted(zip(label_idx, ohe_idx, ohe_idx[1:]), reverse=True)
        cols_in = map(lambda c: [c], cols_in)

        for i, j, k in zipped:
            mask = np.logical_and(cols_out >= j, cols_out < k)
            cols_in.insert(i + 1, cols_out[mask] + n_cols_in)

        cols_in = list(itertools.chain(*cols_in))
        df_out = df_out.iloc[:, cols_in]

        if self.delete:
            df_out = df_out.drop(self.labels, axis=1)

        return df_out

    def _transform_impute(self, df):
        '''
        Imputes ignored and missing values prior to LabelEncoder transform.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.
        '''
        n = len(df)

        for col in self.labels:
            classes = self._le[col].classes_
            n_unq = self._n_unq[col]

            missing = df[col].isnull()
            ignore = ~df[col].isin(set(classes[:n_unq]))
            ignore = np.logical_or(ignore, df[col].isin(set(classes[n_unq:])))

            df.loc[ignore, col] = self._impute_ignore[col]
            df.loc[missing, col] = self._impute_missing[col]

    def fit(self, df, y=None):
        '''
        Fits OneHotLabelEncoder to df.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting OneHotLabelEncoder.

        y: optional
            Passthrough for Pipeline compatibility.

        Returns
        -------
        self
        '''
        df = df.copy()
        n_values = []

        if len(self.labels) == 0:
            self._get_categorical(df)

        self._fit_reduce_cardinality(df, self.labels, self.top)
        self._fit_impute(df)

        for col in self.labels:
            n_unq = self._n_unq[col]
            v_ignore = self._impute_ignore[col]
            v_miss = self._impute_missing[col]

            # specify number of classes to expect for OneHotEncoder
            n_values.append(n_unq + 2)

            enc = self._le[col]
            enc.fit(df[col])  # fit LabelEncoder for column

            # overwrite classes_ attribute with imputation labels
            enc.classes_ = np.hstack([enc.classes_[:n_unq], v_ignore, v_miss])
            # transform LabelEncoder prior to fitting OneHotEncoding
            df[col] = enc.transform(df[col])

        if len(self.labels) == 1:
            df = df[self.labels].values.reshape(-1, 1)
        else:
            df = df[self.labels]

        self._ohe = OneHotEncoder(n_values=n_values, sparse=False)
        self._ohe.fit(df)

        return self

    def transform(self, df):
        '''
        Performs OneHotLabelEncoder transform operation on the input DataFrame.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed.

        Returns
        -------
        df_out: DataFrame, shape = [n_samples, n_features + encoded_features]
            Original DataFrame with OneHotLabelEncoder columns.
        '''
        df_out = df.copy()
        self._transform_impute(df_out)

        for col in self.labels:
            df_out[col] = self._le[col].transform(df_out[col])

        if len(self.labels) == 1:
            df_out = df_out[self.labels].values.reshape(-1, 1)
        else:
            df_out = df_out[self.labels]

        df_out = pd.DataFrame(self._ohe.transform(df_out), index=df.index)
        df_out = self._name_columns(df_out).astype(int)
        df_out = self._select_columns(df, df_out)

        return df_out
