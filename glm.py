from sklearn.linear_model import GammaRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from data_loader import DataSet
from plotting import plot_predictions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--glmtype', choices=['gamma', 'tweedie'], default='gamma')
parser.add_argument('-a', '--alpharange', default="(-4,0,9)", type=str)
parser.add_argument('-p', '--powerrange', default='(1.5,3,7)', type=str)
parser.add_argument('-n', '--nfolds', default=5, type=int)
args = parser.parse_args()

def nloss(y, yh):
    '''
    Mean squared loss. 
    Input: y 1xN: ground truth labels
           yh 1xN: neural network output after Relu 
    Return: MSE 1x1: loss value 
    '''
    diff = np.square(y - yh)
    t = np.sum(diff)
    N = yh.size
    mse = t / (2 * N)

    return mse

def make_param_grid(glmtype, alpharange, powerrange):
    """Prep param grid for GridSearchCV
    """
    alpha_spacing = eval(alpharange)
    assert isinstance(alpha_spacing, (tuple, list)) and (len(alpha_spacing) <= 3), \
        "Invalid `--alpharange` input"
    param_grid = {
        'alpha': np.logspace(*alpha_spacing)
    }
    if glmtype == 'tweedie':
        power_spacing = eval(powerrange)
        assert isinstance(power_spacing, (tuple, list)) and (len(power_spacing) <= 3), \
            "Invalid `--powerrange` input"
        param_grid['power'] = np.linspace(*power_spacing)
    return param_grid

def fit_model(glmtype, param_grid, X, y, cv_fold):
    model = GammaRegressor() if glmtype == 'gamma' else TweedieRegressor()
    gscv = GridSearchCV(model, param_grid, cv=cv_fold)
    gscv.fit(X, y)
    return gscv.best_estimator_

# Load data
ds = DataSet(normalize_y=False)
x = ds.census
y = ds.zillow.flatten()
y /= y.max()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

param_grid = make_param_grid(args.glmtype, args.alpharange, args.powerrange)

# Fit model
model = fit_model(args.glmtype, param_grid, x_train, y_train, args.nfolds)

print(f'Training score: {model.score(x_train, y_train)}')
print(f'Testing score: {model.score(x_test, y_test)}')
print(f'Coefs: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Parameters: {model.get_params()}')

pred = model.predict(x_test)

# Evaluations
loss = nloss(pred, y_test)
print(f'MSE Loss: {loss}')
r2 = r2_score(y_test, pred)
print(f'R2: {r2}')

# Plot predicitons
plot_data = {
    'test': y_test,
    'pred': pred
}
plot_predictions(plot_data, title='Gamma Regression: Predicted vs Actual', filename='glm', zillow_actual=True)
plt.show()