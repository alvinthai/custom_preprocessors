# custom_preprocessors
custom preprocessors for sklearn

## Encoders

- **OneHotLabelEncoder** [Tutorial](http://nbviewer.jupyter.org/github/alvinthai/custom_preprocessors/blob/master/tutorials/OneHotLabelEncoder_Tutorial.ipynb)  
  Automagically runs LabelEncoder and OneHotEncoder on categorical features in a pandas DataFrame. Robust to missing and unknown labels, and has advanced options for filtering out less common labels.

- **CountEncoder** [source](https://github.com/alvinthai/custom_preprocessors/blob/master/encoders/encoders.py#L24)  
  Encoder for reducing cardinality of categorical variables by replacement with their count in the train set.

- **MultinomialNBEncoder** [source](https://github.com/alvinthai/custom_preprocessors/blob/master/encoders/encoders.py#L84)  
  Encoder for reducing cardinality of text data by training a Multinomial Naive Bayes classifier against target labels.

- **TargetEncoder** [source](https://github.com/alvinthai/custom_preprocessors/blob/master/encoders/encoders.py#L150)  
  Encoder for reducing cardinality of categorical variables by replacement with their aggregated category statistic (mean, median, minimum, maximum, standard deviation, sum, or quantile) in the train set. To handle insufficient sample sizes, aggregated statistic could optionally be smoothed with the aggregated prior of entire train set.

## Feature Selection

- **ColumnFilter** [source](https://github.com/alvinthai/custom_preprocessors/blob/master/feature_selection/feature_selection.py#L19)  
  Transformer for automatically filtering columns of a pandas DataFrame, allowing friendly handling of data to pass into scikit-learn estimators that are incompatible with text and time data.

- **GreedyForwardSelection** [source](https://github.com/alvinthai/custom_preprocessors/blob/master/feature_selection/feature_selection.py#L108)  
  Transformer for choosing features with Forward Selection. Forward selection is an iterative method in which we start with having no features in the model. In each iteration, we keep adding the feature which best improves our model until an addition of a new variable does not improve the ROC AUC score of the model for a specified number of iteration rounds.
