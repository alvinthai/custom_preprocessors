from __future__ import division
from _monkeypatch_sklearn import monkeypatch_fit

from collections import defaultdict
from copy import copy
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.model_selection._split import _BaseKFold
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import types


class BaseMultiMetricSearchCV(object):
    '''
    Base class for extending sklearn GridSearchCV and RandomizedSearchCV to
    (1) Report cv_results_ attribute as DataFrame sorted by metric of interest.
    (2) Accomodate reduction in transform steps for pipeline estimators.
    '''
    def binary_params(self, scoring=None, show_train=False, show_std=False,
                      col_sort=None, result_sort=None,
                      transform_before_grid=False, pipeline_split_idx=None,
                      return_train_score='warn'):
        self.show_train = show_train & bool(return_train_score)
        self.show_std = show_std
        self.transform_before_grid = transform_before_grid
        self.pipeline_split_idx = pipeline_split_idx

        if scoring is None:
            scoring = {'roc_auc': 'roc_auc',
                       'accuracy': 'accuracy',
                       'precision': 'precision',
                       'recall': 'recall',
                       'f1': 'f1'}

            if col_sort is None:
                self.col_sort = ['roc_auc', 'accuracy', 'precision',
                                 'recall', 'f1']
            else:
                self.col_sort = col_sort

            if result_sort is None:
                self.result_sort = 'roc_auc'
            else:
                self.result_sort = result_sort
        else:
            self.col_sort = col_sort
            self.result_sort = result_sort

        return scoring

    def fit(self, X, y, groups=None, **fit_params):
        return monkeypatch_fit(self, X, y, groups, **fit_params)

    def fit_report(self, X, y=None, groups=None, **fit_params):
        if 'head' in fit_params:
            head = fit_params.pop('head')
        else:
            head = None

        if 'ascending' in fit_params:
            ascending = fit_params.pop('ascending')
        else:
            ascending = None

        self.fit(X, y, groups, **fit_params)
        return self.report(head, ascending)

    def report(self, head=None, ascending=False):
        assert hasattr(self, 'cv_results_')

        params = pd.DataFrame(self.cv_results_['params'])
        results = pd.DataFrame(self.cv_results_).drop('params', axis=1)

        if self.col_sort is not None:
            metrics = self.col_sort
        elif type(self.scoring) == dict:
            metrics = self.scoring.keys()
        else:
            metrics = ['score']
            self.result_sort = 'score'

        param_cols = filter(lambda x: x.startswith('param_'),
                            self.cv_results_.keys())
        param_cols = sorted(param_cols)
        score_cols = ['mean_test_']

        if self.show_std:
            score_cols.append('std_test_')

        if self.show_train:
            score_cols.append('mean_train_')

        if self.show_train and self.show_std:
            score_cols.append('std_train_')

        cols = param_cols + [s + m for m in metrics for s in score_cols]
        data = pd.DataFrame(self.cv_results_)[cols]

        if self.result_sort is not None:
            sort_by = 'mean_test_{}'.format(self.result_sort)
            data = data.sort_values(sort_by, ascending=ascending)

        return data.head(head)


class BinaryGridSearchCV(BaseMultiMetricSearchCV, GridSearchCV):
    '''
    Extend GridSearchCV with BaseMultiMetricSearchCV.
    Sets default evaluation parameters to report ROC AUC, Accuracy, Precision,
    Recall, and F1 for each set of search hyperparameters.
    '''
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=False, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score="warn", show_train=False, show_std=False,
                 col_sort=None, result_sort=None, transform_before_grid=False,
                 pipeline_split_idx=None):
        assert sklearn.__version__ == '0.19.1'

        self.param_grid = param_grid
        scoring = self.binary_params(scoring, show_train, show_std, col_sort,
                                     result_sort, transform_before_grid,
                                     pipeline_split_idx, return_train_score)

        super(GridSearchCV, self).__init__(estimator, scoring, fit_params,
                                           n_jobs, iid, refit, cv, verbose,
                                           pre_dispatch, error_score,
                                           return_train_score)


class BinaryRandomizedSearchCV(BaseMultiMetricSearchCV, RandomizedSearchCV):
    '''
    Extend RandomizedSearchCV with BaseMultiMetricSearchCV.
    Sets default evaluation parameters to report ROC AUC, Accuracy, Precision,
    Recall, and F1 for each set of search hyperparameters.
    '''
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=False, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score="warn",
                 show_train=False, show_std=False, col_sort=None,
                 result_sort=None, transform_before_grid=False,
                 pipeline_split_idx=None):
        assert sklearn.__version__ == '0.19.1'

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        scoring = self.binary_params(scoring, show_train, show_std, col_sort,
                                     result_sort, transform_before_grid,
                                     pipeline_split_idx, return_train_score)

        super(RandomizedSearchCV, self).__init__(estimator, scoring,
                                                 fit_params, n_jobs, iid,
                                                 refit, cv, verbose,
                                                 pre_dispatch, error_score,
                                                 return_train_score)


class DoubleThresholdCV(object):
    '''
    Custom class for evaluating precision/recall/f1/volume for a range of
    secondary thresholds below a higher threshold with cross-validation model
    training.

    Precision/recall/f1 for values above high threshold are for positive class.
    Precision/recall/f1 for values between high and medium threshold are also
        for positive class.
    Precision/recall/f1 for values below medium threshold are for negative
        class.

    Overall results are plotted by calling the plot_thresholds method after
    fitting the training data.

    This class is also coded to optionally support cross-validation of pipeline
    estimators that depend on target variables for fitting.
    '''
    def __init__(self, estimator, probas, cv=None, high_thld=0.5,
                 pipeline_split_idx=None):
        self.probas = probas
        self.high_thld = high_thld
        self.pipeline_split_idx = pipeline_split_idx

        self.cv_is_func = False
        self.pipeline = None

        if cv is None:
            self.cv = StratifiedKFold(n_splits=4)
        elif type(cv) == int and cv >= 2:
            self.cv = StratifiedKFold(n_splits=cv)
        elif type(cv) in [list, types.GeneratorType]:
            self.cv = cv
        elif isinstance(cv, _BaseKFold):
            self.cv_is_func = True
            self.cv = None
        else:
            raise TypeError('Bad input for cv parameter.')

        if isinstance(estimator, Pipeline):
            pre_pipe_steps = estimator.steps[:pipeline_split_idx]
            new_pipe_steps = estimator.steps[pipeline_split_idx:]
            self.pipeline = Pipeline(pre_pipe_steps)

            if len(new_pipe_steps) == 1:
                self.estimator = new_pipe_steps[0][1]
            else:
                self.estimator = Pipeline(new_pipe_steps)
        else:
            self.estimator = estimator

    @staticmethod
    def _false_cnt(y, y_proba, lte=1, gt=0.5):
        if gt == 0:
            subset = y_proba[:, 1] <= lte
        else:
            subset = (y_proba[:, 1] <= lte) & (y_proba[:, 1] > gt)
        return (y[subset] == 0).sum()

    @staticmethod
    def _true_cnt(y, y_proba, lte=1, gt=0.5):
        if gt == 0:
            subset = y_proba[:, 1] <= lte
        else:
            subset = (y_proba[:, 1] <= lte) & (y_proba[:, 1] > gt)
        return (y[subset] == 1).sum()

    @staticmethod
    def _f1_safe(precision, recall):
        if precision == recall == 0:
            return 0
        else:
            f1 = 2*precision*recall / (precision+recall)
            return f1

    def _calculate_f1(self, precisions, recalls):
        return [self._f1_safe(x, y) for x, y in zip(precisions, recalls)]

    def _calculate_precision_recall_f1_high(self):
        tphigh = sum(self.cv_results_['tphigh'])
        fphigh = sum(self.cv_results_['fphigh'])

        tphigh, fphigh = int(round(tphigh)), int(round(fphigh))

        precision_high = tphigh / (tphigh + fphigh)
        recall_high = tphigh / self.y_pos
        f1_high = self._f1_safe(precision_high, recall_high)
        pct_high = (tphigh + fphigh) / self.y_len

        return [precision_high, recall_high, f1_high, pct_high]

    def _make_threshold_dict(self):
        self.thld_dict = defaultdict(dict)

        for key in self.cv_results_:
            lbl = key.split('@')

            if len(lbl) == 2:
                thld = float(lbl[1])
                metric = lbl[0]
                val = sum(self.cv_results_[key])

                self.thld_dict[thld][metric] = val

        self.thresholds = []
        precision_med, recall_med, pct_med = [], [], []
        precision_low, recall_low, pct_low = [], [], []

        for key in sorted(self.thld_dict.keys()):
            dic = self.thld_dict[key]
            self.thresholds.append(key)

            precision_med.append(dic['tpmed']/(dic['tpmed'] + dic['fpmed']))
            precision_low.append(dic['tplow']/(dic['tplow'] + dic['fplow']))

            recall_med.append(dic['tpmed']/self.y_pos)
            recall_low.append(dic['tplow']/self.y_neg)

            pct_med.append((dic['tpmed'] + dic['fpmed'])/self.y_len)
            pct_low.append((dic['tplow'] + dic['fplow'])/self.y_len)

        self.precision_med = precision_med
        self.precision_low = precision_low

        self.recall_med = recall_med
        self.recall_low = recall_low

        self.f1_med = self._calculate_f1(precision_med, recall_med)
        self.f1_low = self._calculate_f1(precision_low, recall_low)

        self.pct_med = pct_med
        self.pct_low = pct_low

    def fit(self, X, y, groups=None, **fit_params):
        self.y_pos = (y == 1).sum()
        self.y_neg = (y == 0).sum()
        self.y_len = self.y_pos + self.y_neg
        self.y_mean = y.mean()

        high = self.high_thld
        cv_results = defaultdict(list)

        if self.cv_is_func:
            splits = self.cv.split(X, y)
        else:
            splits = self.cv

        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X)

        sw_label, sample_weight = '', None

        for pname in fit_params.keys():
            step, param = pname.split('__', 1)

            if param == 'sample_weight':
                sw_label = pname
                sample_weight = fit_params.pop(pname)

        for (train_idx, test_idx) in splits:
            est = clone(self.estimator)
            cv_fit_params = copy(fit_params)

            if isinstance(X, pd.DataFrame):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            else:
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

            if sample_weight is not None:
                if isinstance(sample_weight, pd.Series):
                    cv_fit_params[sw_label] = sample_weight.iloc[train_idx]
                else:
                    cv_fit_params[sw_label] = sample_weight[train_idx]

            est.fit(X_train, y_train, **cv_fit_params)

            y_probas = est.predict_proba(X_test)

            tphigh = self._true_cnt(y_test, y_probas, gt=high)
            fphigh = self._false_cnt(y_test, y_probas, gt=high)

            cv_results['tphigh'].append(tphigh)
            cv_results['fphigh'].append(fphigh)

            for val in self.probas:
                tpmed = self._true_cnt(y_test, y_probas, lte=high, gt=val)
                fpmed = self._false_cnt(y_test, y_probas, lte=high, gt=val)
                tplow = self._false_cnt(y_test, y_probas, lte=val, gt=0)
                fplow = self._true_cnt(y_test, y_probas, lte=val, gt=0)

                cv_results['tpmed@{}'.format(val)].append(tpmed)
                cv_results['fpmed@{}'.format(val)].append(fpmed)
                cv_results['tplow@{}'.format(val)].append(tplow)
                cv_results['fplow@{}'.format(val)].append(fplow)

        self.cv_results_ = cv_results

        return self

    def plot_thresholds(self):
        self._make_threshold_dict()

        sns.set()
        f, axes = plt.subplots(ncols=2, figsize=(16, 1))

        actual = pd.DataFrame([self.y_mean, 1-self.y_mean], columns=[''],
                              index=['  actual_positive', 'actual_negative'])

        for i, ax in enumerate(axes):
            sns.heatmap(actual.iloc[i:i+1], annot=True, fmt=".1%",
                        linewidths=.5, ax=ax, vmin=0, vmax=1,
                        cbar=False, annot_kws={"size": 14}, square=True)
            ax.yaxis.set_tick_params(rotation=0, labelsize=12)

        f, axes = plt.subplots(ncols=4, figsize=(16.7, 1))

        high_data = self._calculate_precision_recall_f1_high()
        high = pd.DataFrame(high_data, columns=[''],
                            index=['  precision_high ', 'recall_high',
                                   'f1_high', '%_traffic_high'])

        for i, ax in enumerate(axes):
            sns.heatmap(high.iloc[i:i+1], annot=True, fmt=".1%",
                        linewidths=.5, ax=ax, vmin=0, vmax=1,
                        cbar=False, annot_kws={"size": 14}, square=True)
            ax.yaxis.set_tick_params(rotation=0, labelsize=12)

        med_low = pd.DataFrame()
        med_low['precision_medium'] = self.precision_med
        med_low['recall_medium'] = self.recall_med
        med_low['f1_medium'] = self.f1_med
        med_low['%_traffic_medium'] = self.pct_med
        med_low['precision_low'] = self.precision_low
        med_low['recall_low'] = self.recall_low
        med_low['f1_low'] = self.f1_low
        med_low['%_traffic_low'] = self.pct_low
        med_low.index = self.thresholds

        med_low = med_low.T

        f, ax = plt.subplots(figsize=(0.54 + 1.16 * med_low.shape[1], 8))
        sns.heatmap(med_low, annot=True, fmt=".1%", linewidths=.5, ax=ax,
                    vmin=0, vmax=1, square=True)
        ax.yaxis.set_tick_params(rotation=0, labelsize=12)

        plt.show()
