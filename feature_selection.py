from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import numpy as np
import pandas as pd

def forward_feature_selection(X, y, model_init=Ridge, n_features=None, percent_features=0.8, n_folds=5, param_grid={}):
    """Take most informative feature until desired number of features reached.
    Will stop prematurely if all features only hurt performance. Assumes higher score is better

    Parameters
    ----------
    X : np.array
        2d array of shape (samples, features)
    y : np.array
        1d array of shape (samples,)
    model_init : function
        Function to initialize model for making predictions.
        Must initialize sklearn estimator
    n_features : int, optional
        Number of features to keep
    percent_features : float, optiona
        Percent of total features to keep. 
        Lower precedence than `n_features`
    n_folds : int, optional
        Number of folds for cross-validation, default 5
    param_grid : dict, optional
        parameter grid for GridSearchCV

    Return
    ------
    np.array
        Array containing indices of features to include
    """
    # Calculate number of features at stopping point
    if n_features is None:
        n_features = round(X.shape[1] * percent_features)
    # initialize list for included features
    include = set()
    # keep track of previous score
    past_score = -1e3
    new_scores = np.full(X.shape[1], np.nan)
    # loop until stopping point reached
    while len(include) < n_features:
        new_scores[:] = np.nan # reset new score values
        for feat in np.arange(X.shape[1]): # loop through features
            if feat in include: # skip already selected features
                continue
            # fit and evaluate model on selected features + new feature
            feature_inds = list(include) + [feat]
            X_filt = X[:, feature_inds]
            gscv = GridSearchCV(model_init(), param_grid, cv=n_folds)
            gscv.fit(X_filt, y)
            new_scores[feat] = gscv.best_score_
        # find best feature
        best_feat = np.nanargmax(new_scores)
        print(new_scores[best_feat])
        if new_scores[best_feat] < past_score: # stop early if no features improve
            break
        include.add(best_feat)
        past_score = new_scores[best_feat]
    return np.array(include)

def backward_feature_selection(X, y, model_init=Ridge, n_features=None, percent_features=0.8, n_folds=5, param_grid={}):
    """Drop least informative feature until desired number of features reached.

    Parameters
    ----------
    X : np.array
        2d array of shape (samples, features)
    y : np.array
        1d array of shape (samples,)
    model_init : function
        function to initialize model for making predictions
    n_features : int, optional
        Number of features to keep
    percent_features : float, optional
        Percent of total features to keep. 
        Lower precedence than `n_features`
    n_folds : int, optional
        Number of folds for cross-validation, default 5
    param_grid : dict, optional
        parameter grid for GridSearchCV

    Return
    ------
    np.array
        Array containing indices of features to include
    """
    # Calculate number of features at stopping point
    if n_features is None:
        n_features = round(X.shape[1] * percent_features)
    n_drop = X.shape[1] - n_features
    # initialize list for included features
    remove = set()
    new_scores = np.full(X.shape[1], np.nan)
    # loop until stopping point reached
    while len(remove) < n_drop:
        new_scores[:] = np.nan
        for feat in np.arange(X.shape[1]):
            if feat in remove:
                continue
            remove_inds = list(remove) + [feat]
            X_filt = X[:, ~np.isin(np.arange(X.shape[1]), remove_inds)]
            gscv = GridSearchCV(model_init(), param_grid, cv=n_folds)
            gscv.fit(X_filt, y)
            new_scores[feat] = gscv.best_score_
        worst_feat = np.nanargmax(new_scores)
        print(new_scores[worst_feat])
        remove.add(worst_feat)
    return np.setdiff1d(np.arange(X.shape[1]), np.array(remove))

def sequential_feature_selection(X, y, direction='forward', model_init=Ridge, n_features=None, percent_features=0.8, n_folds=5, param_grid={}):
    """Perform sequential feature selection using sklearn built-ins

    Parameters
    ----------
    X : np.array or pd.DataFrame
        2d array of shape (samples, features)
    y : np.array or pd.DataFrame
        1d array of shape (samples,)
    direction : {'forward', 'backward'}, optional
        Whether to do forward or backward feature selection
    model_init : function
        function to initialize model for making predictions
    n_features : int, optional
        Number of features to keep
    percent_features : float, optional
        Percent of total features to keep. 
        Lower precedence than `n_features`
    n_folds : int, optional
        Number of folds for cross-validation, default 5
    param_grid : dict, optional
        parameter grid for GridSearchCV

    Return
    ------
    np.array
        Array containing indices of features to include
    """
    if n_features is None:
        n_features = round(X.shape[1] * percent_features)
    sfs = SequentialFeatureSelector(GridSearchCV(model_init(), param_grid, cv=n_folds), direction=direction, n_features_to_select=n_features).fit(X, y)
    return np.nonzero(sfs.get_support())[0]

def lasso_feature_selection(X, y, threshold=-np.inf, max_features=None, max_percent_features=None, lassocv_params={}):
    """Feature selection using Lasso regression

    Parameters
    ----------
    X : np.array or pd.DataFrame
        2d array of shape (samples, features)
    y : np.array or pd.DataFrame
        1d array of shape (samples,)
    threshold : float, optional
        Minimum magnitude of lasso coefficient to keep.
        Set to -np.inf to use max_features only
    max_features : int, optional
        Maximum number of features to keep. Default None
        sets no limit
    max_percent_features : float, optional
        Maximum percent of original features to keep.
        Default None sets no limit. Lower precedence
        than `max_features`
    lassocv_params : dict, optional
        params for LassoCV
    
    Return
    ------
    np.array
        Array containing indices of features to include
    """
    if max_features is None and max_percent_features is not None:
        max_features = round(X.shape[1] * max_percent_features)
    sfm = SelectFromModel(LassoCV(**lassocv_params), threshold=threshold, max_features=max_features).fit(X, y)
    return np.nonzero(sfm.get_support())[0]

if __name__ == "__main__":
    from data_loader import DataSet

    print('Loading data...')
    ds = DataSet()
    X = ds.census
    y = ds.zillow.flatten()

    print('Starting forward selection...')
    forwardResults = sequential_feature_selection(X, y, direction='forward', model_init=Ridge, param_grid={'alpha': np.logspace(-3, 1, 3)})
    forwardSelections = [ds.dataColumns[i] for i in forwardResults.tolist()]
    print('Column indices:', forwardResults)
    print('Column selections:', forwardSelections)

    print('Starting backward selection...')
    backwardResults = sequential_feature_selection(X, y, direction='backward', model_init=Ridge, param_grid={'alpha': np.logspace(-3, 1, 3)})
    backwardSelections = [ds.dataColumns[i] for i in backwardResults.tolist()]
    print('Column indices:', backwardResults)
    print('Column selections:', backwardSelections)

    print('Starting Lasso feature selection')
    lassoResults = lasso_feature_selection(X, y, max_percent_features=0.8, lassocv_params={'alphas': np.logspace(-3, 1, 5)})
    lassoSelections = [ds.dataColumns[i] for i in lassoResults.tolist()]
    print('Column indices:', lassoResults)
    print('Column selections:', lassoSelections)