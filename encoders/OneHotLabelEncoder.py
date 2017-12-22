import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class OneHotLabelEncoder(BaseEstimator, TransformerMixin):
    '''
    A class for performing a label encoding and one hot encoding sequentially.
    Can handle numerical, categorical, and missing inputs. Also recodes any
    unknown inputs found in transform step as no found value for one hot
    encoding. Fit and transform objects MUST be pandas DataFrame.

    Parameters
    ----------
    label: str
        Column in DataFrame to perform One Hot Label Encoding.
    delete: boolean, default: True
        Whether to delete the label column from DataFrame after transformation
    output_missing_col: boolean, default: False.
        Whether to output an indicator column for missing values, should
        specify to True if expecting missing values.
    missing_fill:
        An indicator value used to fill in label encoding when values are
        missing, should be different than unique values in input column.
    '''
    def __init__(self, label, delete=True, output_missing_col=False,
                 missing_fill=-999999.0):
        self._le = LabelEncoder()
        self._ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        self.label = label
        self.delete = delete
        self.output_missing_col = output_missing_col
        self.missing_fill = missing_fill

    def _delete_columns(self, X, data):
        '''
        Utility function for deleting label column from input DataFrame and
        missing column for OneHotEncoded values in the output DataFrame. The
        deletion of these columns is controlled by the init parameters.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Original DataFrame before OneHotLabelEncoding.
        data: DataFrame, shape = [n_samples, n_classes + 1]
            OneHotEncodedDataFrame with extra column as indicator for missing
            values.
        '''
        if self.delete:
            del X[self.label]

        if not self.output_missing_col:
            if 'missing_' + self.label in data.columns:
                del data['missing_' + self.label]

    def _label_transform(self, classes, ncols):
        '''
        Function for outputing column names for one hot encoding numpy array.
        Prepends column name to unique value from OneHotLabelEncoding.

        Parameters
        ----------
        classes: list
            List of unique values from OneHotLabelEncoder column.
        ncols: int
            Number of columns in OneHotLabelEncoder DataFrame. If equals to
            n_classes + 1, there is an indicator column for missing values.

        Returns
        -------
        output: list of str
            List of column names for OneHotLabelEncoder DataFrame.
        '''
        classes = np.char.array(classes)
        output = []

        for x in classes:
            if str(self.missing_fill) == x:
                output.append('missing_' + self.label)
            else:
                output.append(self.label + '_' + x)

        # Transform DataFrame has missing values and fit DataFrame does not
        if len(classes) == ncols - 1:
                output.append('missing_' + self.label)

        return output

    def _na_transform(self, y):
        '''
        Turns missing values to an indicator value (i.e. -999999) prior to
        label transform.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            Input values from column to be OneHotLabelEncoder transformed.

        Returns
        -------
        y: array-like, shape = [n_samples, ]
            Input values with missing values converted to a spurious value.
        '''
        mask = y.isnull()

        if mask.sum():
            if self.missing_fill in y.unique():
                raise AssertionError('NA fill error. Please initialize class '
                                     'with a missing_fill value not found in '
                                     'column.')

            if y.dtype == object:
                y.loc[mask] = str(self.missing_fill)
            elif y.dtype in (int, float):
                y.loc[mask] = self.missing_fill

        return y

    def _output_missing_col(self, data, ncols, missing_mask, classes):
        '''
        Adds indicator column for missing values.

        Parameters
        ----------
        data: DataFrame, shape = [n_samples, n_encoded_features]
            DataFrame with a column for each unique value found in the
            OneHotLabelEncoder fitting step.
        ncols: int
            Count of unique values in OneHotLabelEncoder fitted column.
        missing_mask: array-like, shape = [n_samples, ]
            Array of True or False labels indicating whether the value was
            missing from the column specified in OneHotLabelEncoder.
        classes: list
            List of unique values from OneHotLabelEncoder fitted columns.

        Returns
        -------
        data: DataFrame, shape = [n_samples, n_encoded_features (+ 1)]
            If self.output_missing_col is True, outputs data with an additional
            column that indicates if the value was missing from the
            OneHotLabelEncoder column.
        ncols: int
            If self.output_missing_col is True, returns the column dimension of
            the OneHotLabelEncoder DataFrame (input ncols + 1).
        '''
        if self.output_missing_col:
            misscol = np.zeros(len(data))

            # Missing values found in transform and not in fit
            if missing_mask is not None:
                misscol[missing_mask] = 1

            # Missing values not found in fit
            if str(self.missing_fill) not in classes:
                # Add column for missing values
                data = np.hstack([data, misscol.reshape(-1, 1)])
                ncols += 1

        return data, ncols

    def _unknown_transform(self, y, unknown):
        '''
        Turns unknown values found in transform into an invalid label encoder
        value. OneHotEncoder will return 0 on all columns with unknown values.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            Input values (with or without unknown values not seen in fit step)
            from column to be OneHotLabelEncoder transformed.
        unknown: set
            A set of unknown values not seen in the fit step.

        Returns
        -------
        y: array-like, shape = [n_samples, ]
            LabelEncoder transformed y values. Unknown values are set to a
            value not seen in the OneHotEncoder fit step:
            len(self._le.classes_).
        '''
        mask = y.isin(unknown)

        # Set unknown values to a valid label
        y[mask] = self._le.inverse_transform(0)
        y = self._le.transform(y)

        # Set unknown values to an invalid class
        y[mask] = len(self._le.classes_)

        return y

    def fit(self, X, y=None):
        '''
        Fits OneHotLabelEncoder to X.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame for fitting the OneHotLabelEncoder transformer.
        y: optional
            Passthrough for Pipeline compatibility.

        Returns
        -------
        self
        '''
        y = self._na_transform(X[self.label].copy())

        data = self._le.fit_transform(y)
        self._ohe.fit(data.reshape(-1, 1))

        return self

    def transform(self, X):
        '''
        Performs OneHotLabelEncoder transform operation on the input DataFrame.

        Parameters
        ----------
        X: DataFrame, shape = [n_samples, n_features]
            Input DataFrame to be transformed. Must include column specified in
            self.label.

        Returns
        -------
        X: DataFrame, shape = [n_samples, n_features + n_encoded_features]
            Original DataFrame with OneHotLabelEncoder columns.
        '''
        y = self._na_transform(X[self.label].copy())

        # Checks if there are unknown classes
        unknown = set(y.unique()) - set(self._le.classes_)

        # Checks if missing values found in transform and fit step did not
        # handle missing values
        if self.missing_fill in unknown:
            missing_mask = y == self.missing_fill
        elif str(self.missing_fill) in unknown:
            missing_mask = y == str(self.missing_fill)
        else:
            missing_mask = None

        data = self._unknown_transform(y, unknown)
        data = self._ohe.transform(data.reshape(-1, 1))

        classes = np.char.array(self._le.classes_)
        ncols = data.shape[1]

        data, ncols = self._output_missing_col(data, ncols,
                                               missing_mask, classes)
        data = pd.DataFrame(data, index=X.index,
                            columns=self._label_transform(classes, ncols),
                            dtype=int)

        self._delete_columns(X, data)
        X = pd.concat([X, data], axis=1)

        return X
