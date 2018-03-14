"""
Feature Selection Transfomers:
    - ColumnFilter
    - GreedyForwardSelection
"""

# Author: Alvin Thai <alvinthai@gmail.com>
# GreedyForwardSelection partially adopted from Abhishek Thakur

from copy import copy
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd


class ColumnFilter(BaseEstimator, TransformerMixin):
    '''
    A class for automatically filtering columns of a pandas DataFrame, allowing
    friendly handling of data to pass into sklearn estimators that are
    incompatible with text and time data. By default, categorical and datetime
    data type columns are excluded from data transformation.

    Parameters
    ----------
    drop_cols: list, optional
        List of columns to exclude in addition to automatic filtered columns.

    filter_cols: list, optional
        List of columns to filter dataset to. If specified, automatic column
        filtering is ignored in data transformation.

    exclude_dtypes: list, optional, default: [np.dtype('object'), np.dtype('<M8[ns]')]
        List of data types to automatically filter from pandas DataFrame.
    '''
    def __init__(self, drop_cols=None, filter_cols=None, exclude_dtypes=None):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols

        if filter_cols is None:
            self.filter_cols = []
        else:
            self.filter_cols = filter_cols

        if exclude_dtypes is None:
            self.exclude_dtypes = [np.dtype('object'), np.dtype('<M8[ns]')]
        else:
            self.exclude_dtypes = exclude_dtypes

        self.exclude = None

    def fit(self, df, y=None):
        '''
        Fits ColumnFilter to df.

        Parameters
        ----------
        df: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting ColumnFilter.

        y: array-like, shape = [n_samples, ], optional
            Passthrough for Pipeline compatibility.

        Returns
        -------
        self
        '''
        self.exclude = set()

        for col in df.columns:
            if df[col].dtype in self.exclude_dtypes:
                # exclude column with unwanted data type
                self.exclude.add(col)
            if col in self.drop_cols:
                # drop columns specified by user
                self.exclude.add(col)

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
        df: DataFrame, shape = [n_samples, filtered_features]
            Original DataFrame with filtered columns.
        '''
        df = df.copy()

        if len(self.filter_cols) > 0:
            # filter to specific columns if self.filter_cols is specified
            return df[self.filter_cols]
        else:
            # filter out excluded columns
            return df.drop(self.exclude, axis=1)


class GreedyForwardSelection(object):
    '''
    A class for choosing features with forward selection. Forward selection is
    an iterative method in which we start with having no features in the model.
    In each iteration, we keep adding the feature which best improves our model
    until an addition of a new variable does not improve the ROC AUC score of
    the model for <window> iteration rounds.

    Parameters
    ----------
    model: estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Estimator needs to provide a predict_proba function.

    verbose: int, optional, default: 0
        Controls fit verbosity: the higher, the more messages are printed.

    window: int, optional, default: 1
        Controls number of iteration rounds to continue forward selection
        until no improvement in ROC AUC score is achieved.

    good_features: list of int, optional, default: None
        Preselected indexes of good features to start with prior to fitting
        GreedyForwardSelection.

    feature_history: list of int, optional, default: None
        Indexes of good_features in the order the features were chosen for
        forward selection.

    score_features: list of float, optional, default: None
        ROC AUC scores for feature_history.
    '''
    def __init__(self, model, verbose=0, window=1, good_features=None,
                 feature_history=None, score_history=None):
        self.model = model
        self.verbose = verbose
        self.window = window
        self.good_features = good_features
        self.feature_history = feature_history
        self.score_history = score_history
        self.logger = []

        # continues GreedyForwardSelection from past results if user specifies
        # good_features, feature_history, score_history
        if good_features is not None:
            self.good_features = good_features
        else:
            self.good_features = []

        if feature_history is not None:
            self.feature_history = feature_history
        elif len(self.good_features) > 0:
            self.feature_history = copy(self.good_features)
        else:
            self.feature_history = []

        if score_history is not None:
            self.score_history = score_history
        elif len(self.good_features) > 0:
            self.score_history = [0] * len(self.good_features)
        else:
            self.score_history = []

    def evaluateScore(self, X_train, y_train, X_eval, y_eval):
        '''
        Evaluates ROC AUC score for the input data.

        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features]
            Input training data for evaulaing ROC AUC.

        y_train: array-like, shape = [n_samples, ]
            True labels for X_train.

        X_eval: array-like, shape = [n_samples, n_features]
            Input for evaluating ROC AUC on unseen data.
            If None, cross validation is performed on X_train.

        y_eval: array-like, shape = [n_samples, ]
            True labels for X_eval.

        Returns
        -------
        self
        '''
        model = clone(self.model)

        if X_eval is not None:
            # evaluate ROC AUC on unseen data
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_eval)[:, 1]
            auc = roc_auc_score(y_eval, predictions)
        else:
            # perform cross validation to calculate ROC AUC
            aucs = cross_val_score(model, X_train, y_train, scoring='roc_auc')
            auc = np.mean(aucs)
        return auc

    def print_log(self):
        '''
        Prints the output log saved during fitting.
        '''
        print '\n'.join(self.logger)

    def selectionLoop(self, X_train, y_train, X_test, y_test, cols):
        '''
        Runs forward feature selection loop for GreedyForwardSelection.
        At each loop, selects the feature that returns highest ROC AUC score.
        Feature loop is repeated until all features are added or ROC AUC score
        does not improve for <self.window> rounds.

        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features]
            Input data for training GreedyForwardSelection.

        y_train: array-like, shape = [n_samples, ]
            True labels for X_train.

        X_test: array-like, shape = [n_samples, n_features]
            Input data for evaluating column filtered datasets.
            If None, cross validation is performed on X_train.

        y_test: array-like, shape = [n_samples, ]
            True labels for X_test.

        cols: list
            List of column names.

        Returns
        -------
        self
        '''
        self.logger = []

        if X_test is not None:
            dset_type = 'test set'
        else:
            dset_type = 'cross val'

        good_features = set(self.good_features)
        feature_history = self.feature_history
        score_history = self.score_history

        num_features = X_train.shape[1]
        window = min(self.window, num_features)
        comp = window + 1

        # loop continues until AUC score does not improve for <window> rounds
        while len(score_history) < comp or \
                score_history[-comp] < max(score_history[-window:]):

            if len(score_history) == num_features:
                break

            scores = []
            for feature in range(num_features):
                if feature not in good_features:
                    selected_features = list(good_features) + [feature]

                    Xts_train = np.column_stack(X_train[:, j] for j in
                                                selected_features)
                    if X_test is not None:
                        Xts_test = np.column_stack(X_test[:, j] for j in
                                                   selected_features)
                    else:
                        Xts_test = None

                    score = round(self.evaluateScore(Xts_train, y_train,
                                                     Xts_test, y_test), 4)
                    scores.append((score, feature))

                    if self.verbose > 1:
                        msg = "Current ROC AUC, f{}: ".format(feature)
                        print msg, np.mean(score)

            good_features.add(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1][0])
            feature_history.append(sorted(scores)[-1][1])

            current_feats = cols[feature_history]
            msg = 'Best ROC AUC with {} features ({}):'
            msg = msg.format(len(score_history), dset_type)

            line1 = '-'*80
            line2 = '\nCurrent Features:\n{}'.format(current_feats)
            line3 = '\n{} {}\n'.format(msg, score_history[-1])

            self.logger.extend([line1, line2, line3])

            if self.verbose > 0:
                print line1
                print line2
                print line3

        best_idx = np.argmax(score_history)

        for i in xrange(best_idx+1, len(feature_history)):
            good_features.remove(feature_history[i])

        good_features = sorted(list(good_features))
        score_history = score_history[:best_idx+1]
        feature_history = feature_history[:best_idx+1]

        line1 = '-'*80
        line2 = '\nNo AUC improvements for {} consectuive rounds.\nStopping...'
        line2 = line2.format(window)
        line3 = '\nSelected {} Features:\n{}'
        line3 = line3.format(len(good_features), cols[good_features])

        self.logger.extend([line1, line2, line3])

        if self.verbose:
            print line1
            print line2
            print line3

        self.good_features = good_features
        self.score_history = score_history
        self.feature_history = feature_history

        return good_features

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        '''
        Fits GreedyForwardSelection to X_train.

        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features]
            Input data for training GreedyForwardSelection.

        y_train: array-like, shape = [n_samples, ]
            True labels for X_train.

        X_test: array-like, shape = [n_samples, n_features], optional
            Input data for evaluating column filtered datasets.
            If not provided, cross validation is performed on X_train.

        y_test: array-like, shape = [n_samples, ], optional
            True labels for X_test.

        Returns
        -------
        self
        '''
        if type(X_train) == pd.core.frame.DataFrame:
            cols = np.array(X_train.columns.tolist())
            X_train, y_train = X_train.values, y_train.values
        else:
            cols = np.arange(X_train.shape[1])

        if X_test is not None and type(X_test) == pd.core.frame.DataFrame:
            assert X_train.shape[1] == X_test.shape[1]
            X_test, y_test = X_test.values, y_test.values

        self.selectionLoop(X_train, y_train, X_test, y_test, cols)

        return self

    def transform(self, X):
        '''
        Performs transform operation on the input data.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data to be transformed.

        Returns
        -------
        X: array-like, shape = [n_samples, selected_features]
            Original data with selected columns.
        '''
        if type(X) == pd.core.frame.DataFrame:
            X = X.copy()
            return X.iloc[:, self.good_features]
        else:
            return X[:, self.good_features]
