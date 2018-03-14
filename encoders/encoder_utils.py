from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import numpy as np


class GridMultinomialNB(BaseEstimator, ClassifierMixin):
    '''
    Performs Multinomial Naive Bayes classification on text data.
    Text data is transformed into bag of words with stop words removed.
    Optimal alpha value for fitting MultinomialNB is found via grid search.
    '''
    def __init__(self):
        nb_grid = {'alpha': [0.1, 0.4, 0.7, 1, 1.3, 1.6, 2.0]}
        self.nb_gridsearch = GridSearchCV(MultinomialNB(), nb_grid,
                                          n_jobs=-1, scoring='roc_auc')

    def fit(self, X, y):
        if len(np.unique(y)) == 2 and np.array(y).dtype in [bool, float, int]:
            class_prior = [1-np.mean(y), np.mean(y)]
            self.nb_gridsearch.param_grid['class_prior'] = [class_prior]

        self.vect = CountVectorizer(stop_words='english')
        self.mat = self.vect.fit_transform(X)
        self.nb_gridsearch.fit(self.mat, y)
        self.model = self.nb_gridsearch.best_estimator_
        return self

    def predict(self, X):
        vect_x = self.vect.transform(X)
        return self.model.predict(vect_x)

    def predict_proba(self, X):
        vect_x = self.vect.transform(X)
        return self.model.predict_proba(vect_x)
