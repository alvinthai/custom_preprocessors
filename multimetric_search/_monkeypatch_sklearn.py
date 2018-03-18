from __future__ import division, print_function
from functools import partial
from itertools import product
from collections import defaultdict
from scipy.stats import rankdata
import numpy as np
import warnings
import time

from sklearn.base import is_classifier, clone
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.externals import six
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (_aggregate_score_dicts,
                                                 _index_param_value, _score)
from sklearn.pipeline import Pipeline
from sklearn.utils.deprecation import DeprecationDict
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import indexable, _num_samples


def monkeypatch_fit(self, X, y=None, groups=None, **fit_params):
    if self.fit_params is not None:
        warnings.warn('"fit_params" as a constructor argument was '
                      'deprecated in version 0.19 and will be removed '
                      'in version 0.21. Pass fit parameters to the '
                      '"fit" method instead.', DeprecationWarning)
        if fit_params:
            warnings.warn('Ignoring fit_params passed as a constructor '
                          'argument in favor of keyword arguments to '
                          'the "fit" method.', RuntimeWarning)
        else:
            fit_params = self.fit_params
    estimator = self.estimator
    cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

    scorers, self.multimetric_ = _check_multimetric_scoring(
        self.estimator, scoring=self.scoring)

    if self.multimetric_:
        if self.refit is not False and (
                not isinstance(self.refit, six.string_types) or
                # This will work for both dict / list (tuple)
                self.refit not in scorers):
            raise ValueError("For multi-metric scoring, the parameter "
                             "refit must be set to a scorer key "
                             "to refit an estimator with the best "
                             "parameter setting on the whole data and "
                             "make the best_* attributes "
                             "available for that metric. If this is not "
                             "needed, refit should be set to False "
                             "explicitly. %r was passed." % self.refit)
        else:
            refit_metric = self.refit
    else:
        refit_metric = 'score'

    X, y, groups = indexable(X, y, groups)
    n_splits = cv.get_n_splits(X, y, groups)
    # Regenerate parameter iterable for each fit
    candidate_params = list(self._get_param_iterator())
    n_candidates = len(candidate_params)
    if self.verbose > 0:
        print("Fitting {0} folds for each of {1} candidates, totalling"
              " {2} fits".format(n_splits, n_candidates,
                                 n_candidates * n_splits))

    base_estimator = clone(self.estimator)
    pre_dispatch = self.pre_dispatch

    # ===================================================================
    # BEGIN MONKEYPATCH MODIFICATION
    # ===================================================================

    parallel_cv = cv.split(X, y, groups)

    if type(self.pipeline_split_idx) == int and isinstance(base_estimator,
                                                           Pipeline):
        split_idx = self.pipeline_split_idx

        pre_pipe_steps = base_estimator.steps[:split_idx]
        new_pipe_steps = base_estimator.steps[split_idx:]
        memory = base_estimator.memory

        pre_pipe = Pipeline(pre_pipe_steps, memory)

        if len(new_pipe_steps) == 1:
            est_name, base_estimator = new_pipe_steps[0]
        else:
            est_name = None
            base_estimator = Pipeline(new_pipe_steps, memory)

        fit_params_pre_pipe = {}
        steps_pre_pipe = [tup[0] for tup in pre_pipe_steps]
        fit_param_keys = fit_params.keys()

        for pname in fit_param_keys:
            step, param = pname.split('__', 1)

            if step in steps_pre_pipe:
                fit_params_pre_pipe[pname] = fit_params.pop(pname)
            elif step == est_name:
                fit_params[param] = fit_params.pop(pname)

        if est_name is not None:
            for dic in candidate_params:
                for k in dic:
                    step, param = k.split('__', 1)

                    if step == est_name:
                        dic.update({param: dic.pop(k)})

        try:
            X = pre_pipe.fit_transform(X, **fit_params_pre_pipe)
        except TypeError:
            raise RuntimeError('Pipeline before pipeline_split_idx requires '
                               'fitting to y. Please initialize with an '
                               'earlier index.')

    if self.transform_before_grid and isinstance(base_estimator, Pipeline):
        pipe = base_estimator
        est_name, base_estimator = pipe.steps.pop()
        X_cv, y_cv, parallel_cv = [], [], []
        sample_count = 0

        fit_params_est = {}
        fit_param_keys = fit_params.keys()

        for pname in fit_param_keys:
            step, param = pname.split('__', 1)
            if step == est_name:
                fit_params_est[param] = fit_params.pop(pname)

        for dic in candidate_params:
            for k in dic:
                step, param = k.split('__', 1)

                if step == est_name:
                    dic.update({param: dic.pop(k)})

        for (train, test) in cv.split(X, y, groups):
            if y is not None:
                if isinstance(X, pd.DataFrame):
                    pipe.fit(X.iloc[train], y.iloc[train], **fit_params)
                else:
                    pipe.fit(X[train], y[train], **fit_params)
                y_cv.append(y)
            else:
                if isinstance(X, pd.DataFrame):
                    pipe.fit(X.iloc[train], **fit_params)
                else:
                    pipe.fit(X[train], **fit_params)

            X_cv.append(pipe.transform(X))

            train = train + sample_count
            test = test + sample_count
            sample_count += len(train)
            sample_count += len(test)

            parallel_cv.append((train, test))

        if isinstance(X, pd.DataFrame):
            X = pd.concat(tuple(X_cv))
        else:
            X = np.vstack(tuple(X_cv))

        if y is not None:
            if isinstance(y, pd.Series):
                y = pd.concat(tuple(y_cv))
            else:
                y = np.hstack(tuple(y_cv))

            if 'sample_weight' in fit_params_est:
                samp_weight = fit_params_est['sample_weight']
                fit_params_est['sample_weight'] = np.tile(samp_weight,
                                                          len(y_cv))

        fit_params = fit_params_est

    out = Parallel(
        n_jobs=self.n_jobs, verbose=self.verbose,
        pre_dispatch=pre_dispatch
    )(delayed(monkeypatch_fit_and_score)
      (clone(base_estimator), X, y, scorers, train,
                              test, self.verbose, parameters,
                              fit_params=fit_params,
                              return_train_score=self.return_train_score,
                              return_n_test_samples=True,
                              return_times=True, return_parameters=False,
                              error_score=self.error_score)
      for parameters, (train, test) in product(candidate_params,
                                               parallel_cv))

    # ===================================================================
    # END MONKEYPATCH MODIFICATION
    # ===================================================================

    # if one choose to see train score, "out" will contain train score info
    if self.return_train_score:
        (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)
    else:
        (test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)

    # test_score_dicts and train_score dicts are lists of dictionaries and
    # we make them into dict of lists
    test_scores = _aggregate_score_dicts(test_score_dicts)
    if self.return_train_score:
        train_scores = _aggregate_score_dicts(train_score_dicts)

    # TODO: replace by a dict in 0.21
    results = (DeprecationDict() if self.return_train_score == 'warn'
               else {})

    def _store(key_name, array, weights=None, splits=False, rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        # When iterated first by splits, then by parameters
        # We want `array` to have `n_candidates` rows and `n_splits` cols.
        array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                          n_splits)
        if splits:
            for split_i in range(n_splits):
                # Uses closure to alter the results
                results["split%d_%s"
                        % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results['mean_%s' % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(np.average((array -
                                         array_means[:, np.newaxis]) ** 2,
                                        axis=1, weights=weights))
        results['std_%s' % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-array_means, method='min'), dtype=np.int32)

    _store('fit_time', fit_time)
    _store('score_time', score_time)
    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(partial(MaskedArray,
                                        np.empty(n_candidates,),
                                        mask=True,
                                        dtype=object))
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)
    # Store a list of param dicts at the key 'params'
    results['params'] = candidate_params

    # NOTE test_sample counts (weights) remain the same for all candidates
    test_sample_counts = np.array(test_sample_counts[:n_splits],
                                  dtype=np.int)
    for scorer_name in scorers.keys():
        # Computed the (weighted) mean and std for test scores alone
        _store('test_%s' % scorer_name, test_scores[scorer_name],
               splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            prev_keys = set(results.keys())
            _store('train_%s' % scorer_name, train_scores[scorer_name],
                   splits=True)

            if self.return_train_score == 'warn':
                for key in set(results.keys()) - prev_keys:
                    message = (
                        'You are accessing a training score ({!r}), '
                        'which will not be available by default '
                        'any more in 0.21. If you need training scores, '
                        'please set return_train_score=True').format(key)
                    # warn on key access
                    results.add_warning(key, message, FutureWarning)

    # For multi-metric evaluation, store the best_index_, best_params_ and
    # best_score_ iff refit is one of the scorer names
    # In single metric evaluation, refit_metric is "score"
    if self.refit or not self.multimetric_:
        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_params_ = candidate_params[self.best_index_]
        self.best_score_ = results["mean_test_%s" % refit_metric][
            self.best_index_]

    if self.refit:
        self.best_estimator_ = clone(base_estimator).set_params(
            **self.best_params_)
        if y is not None:
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            self.best_estimator_.fit(X, **fit_params)

    # Store the only scorer not as a dict for single metric evaluation
    self.scorer_ = scorers if self.multimetric_ else scorers['score']

    self.cv_results_ = results
    self.n_splits_ = n_splits

    return self


def monkeypatch_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                              parameters, fit_params, return_train_score=False,
                              return_parameters=False, return_n_test_samples=False,
                              return_times=False, return_estimator=False,
                              error_score='raise-deprecating'):
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    # ===================================================================
    # BEGIN MONKEYPATCH MODIFICATION
    # ===================================================================

    try:
        if isinstance(estimator, Pipeline):
            pipe = estimator
            est_name, estimator = pipe.steps.pop()

            fit_params_est = {}
            fit_param_keys = fit_params.keys()

            for pname in fit_param_keys:
                step, param = pname.split('__', 1)
                if step == est_name:
                    fit_params_est[param] = fit_params.pop(pname)

        else:
            pipe = None

        if y_train is None:
            if pipe is not None:
                X_train = pipe.fit_transform(X_train, **fit_params)
                X_test = pipe.transform(X_test, **fit_params)
                fit_params = fit_params_est
            estimator.fit(X_train, **fit_params)
        else:
            if pipe is not None:
                X_train = pipe.fit_transform(X_train, y_train, **fit_params)
                X_test = pipe.transform(X_test, **fit_params)
                fit_params = fit_params_est
            estimator.fit(X_train, y_train, **fit_params)

    # ===================================================================
    # END MONKEYPATCH MODIFICATION
    # ===================================================================

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                   [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                        [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret
