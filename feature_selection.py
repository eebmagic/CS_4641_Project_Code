from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

if __name__ == "__main__":
    from data_loader import DataSet
    ds = DataSet()
    data_columns = ['TotalPop', 'Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific', 'Citizen', 'Income',
       'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment']
    X = ds.census[data_columns].to_numpy()

    censusCounties = set(ds.census['Full County Name'])
    selection = ds.zillow['Full County Name'].isin(censusCounties)
    ds.zillow = ds.zillow[selection]
    month = 'December 2015'
    y = ds.zillow[month].to_numpy()

    X = StandardScaler().fit_transform(X)
    print(forward_feature_selection(X, y, model_init=SVR, param_grid={}))
    print(backward_feature_selection(X, y, model_init=SVR, param_grid={}))