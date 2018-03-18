from __future__ import division
from _monkeypatch_sklearn import monkeypatch_fit

from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn


class BaseMultiMetricSearchCV(object):
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


class DoubleThresholdCV(GridSearchCV):
    def __init__(self, estimator, probas, param_grid=None, n_jobs=1, iid=True,
                 refit=False, cv=None, verbose=0, transform_before_grid=False,
                 pipeline_split_idx=None, high_thld=0.5):
        assert sklearn.__version__ == '0.19.1'

        self.probas = probas
        self.transform_before_grid = transform_before_grid
        self.pipeline_split_idx = pipeline_split_idx
        self.high_thld = high_thld

        self.scoring = {}
        high = high_thld
        true_cnt = self._true_cnt
        false_cnt = self._false_cnt

        if param_grid is None:
            self.param_grid = {}
        else:
            self.param_grid = param_grid

        for val in self.probas:
            tp_high = make_scorer(true_cnt, needs_proba=True, lte=1, gt=high)
            fp_high = make_scorer(false_cnt, needs_proba=True, lte=1, gt=high)
            tp_med = make_scorer(true_cnt, needs_proba=True, lte=high, gt=val)
            fp_med = make_scorer(false_cnt, needs_proba=True, lte=high, gt=val)
            tp_low = make_scorer(false_cnt, needs_proba=True, lte=val, gt=0)
            fp_low = make_scorer(true_cnt, needs_proba=True, lte=val, gt=0)

            val_dict = {'#tphigh'.format(val): tp_high,
                        '#fphigh'.format(val): fp_high,
                        '#tpmed@{}'.format(val): tp_med,
                        '#fpmed@{}'.format(val): fp_med,
                        '#tplow@{}'.format(val): tp_low,
                        '#fplow@{}'.format(val): fp_low}

            self.scoring.update(val_dict)

        super(GridSearchCV, self).__init__(estimator, self.scoring, iid=iid,
                                           fit_params=None, n_jobs=n_jobs,
                                           refit=refit, cv=cv, verbose=verbose,
                                           return_train_score=False)

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
        tphigh = self.cv_results_['mean_test_#tphigh'] * self.n_splits_
        fphigh = self.cv_results_['mean_test_#fphigh'] * self.n_splits_

        tphigh, fphigh = int(round(tphigh)), int(round(fphigh))

        precision_high = tphigh / (tphigh + fphigh)
        recall_high = tphigh / self.y_pos
        f1_high = self._f1_safe(precision_high, recall_high)
        pct_high = (tphigh + fphigh) / self.y_len

        return [precision_high, recall_high, f1_high, pct_high]

    def _make_threshold_dict(self):
        self.thld_dict = defaultdict(dict)

        for key in self.cv_results_:
            lbl = key.replace('#', '@').split('@')

            if len(lbl) == 3 and lbl[0] == 'mean_test_':
                thld = float(lbl[2])
                metric = lbl[1]

                val = self.cv_results_[key] * self.n_splits_
                self.thld_dict[thld][metric] = int(round(val))

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

        return monkeypatch_fit(self, X, y, groups, **fit_params)

    def plot_thresholds(self):
        self._make_threshold_dict()

        sns.set()
        f, axes = plt.subplots(ncols=4, figsize=(16.7, 1))

        high_data = self._calculate_precision_recall_f1_high()
        high = pd.DataFrame(high_data, columns=[''],
                            index=['precision_high', 'recall_high',
                                   'f1_high', '%_traffic_high'])

        for i, ax in enumerate(axes):
            sns.heatmap(high.iloc[i:i+1], annot=True, fmt=".1%",
                        linewidths=.5, ax=ax, vmin=0, vmax=1,
                        cbar=False, annot_kws={"size": 14}, square=True)
            ax.yaxis.set_tick_params(rotation=0, labelsize=14)

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

        f, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(med_low, annot=True, fmt=".1%", linewidths=.5, ax=ax,
                    vmin=0, vmax=1)
        ax.yaxis.set_tick_params(rotation=0, labelsize=12)
        plt.show()
