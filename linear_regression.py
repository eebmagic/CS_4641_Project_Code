import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from data_loader import DataSet
from plotting import plot_predictions
 
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


def avg_r2(x, y, iters=300, getcoefs=False):
    scores = []
    coefs = []
    for _ in range(iters):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        reg = LinearRegression().fit(x_train, y_train)
        pred = reg.predict(x_test)

        r2 = r2_score(y_test, pred)
        scores.append(r2)

        if getcoefs:
            coefs.append(reg.coef_)

    scores = np.array(scores)
    coefs = np.array(coefs)
    coefs = np.average(coefs, axis=0)

    if getcoefs:
        return np.average(scores), np.std(scores), coefs
    else:
        return np.average(scores), np.std(scores)


# Load data
ds = DataSet()
x = ds.census
y = ds.zillow.flatten()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit model
reg = LinearRegression().fit(x_train, y_train)

print(f'Training score: {reg.score(x_train, y_train)}')
print(f'Testing score: {reg.score(x_test, y_test)}')
print(f'Coefs: {reg.coef_}')
print(f'Intercept: {reg.intercept_}')

# Make predictions
pred = reg.predict(x_test)

# Evalutations
loss = nloss(pred, y_test)
print(f'MSE Loss: {loss}')
r2 = r2_score(y_test, pred)
print(f'R2: {r2}')

# Plot predicitons
plot_data = {
    'test': y_test,
    'pred': pred
}
plot_predictions(plot_data, title='Linear Regression: Predicted vs Actual', filename='linearreg')

total_iters = 10_000
normal_average = avg_r2(x, y, iters=total_iters, getcoefs=True)


### AGAIN, but with feature reduction
# Feature reduction with hardcoded values from previous results
forwardResults = np.array([3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 26, 27, 29, 30, 33])
x = x[:, forwardResults]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit model
reg = LinearRegression().fit(x_train, y_train)

print(f'Training score: {reg.score(x_train, y_train)}')
print(f'Testing score: {reg.score(x_test, y_test)}')
print(f'Coefs: {reg.coef_}')
print(f'Intercept: {reg.intercept_}')

# Make predictions
pred = reg.predict(x_test)

# Evalutations
loss = nloss(pred, y_test)
print(f'MSE Loss: {loss}')
r2 = r2_score(y_test, pred)
print(f'R2: {r2}')

# Plot predicitons
plot_data = {
    'test': y_test,
    'pred': pred
}

title = 'Linear Regression with Forward Selected Features: Predicted vs Actual'
filename = 'linearForward'
plot_predictions(plot_data, title=title, filename=filename)

feature_reduced_average = avg_r2(x, y, iters=total_iters, getcoefs=True)

print(f'\nAverage R2 value with all data: {normal_average[0]}')
print(f'Average R2 value with feature reduction: {feature_reduced_average[0]}')
print(f'R2 std dev with all data: {normal_average[1]}')
print(f'R2 std dev with feature reduction: {feature_reduced_average[1]}')

plt.show()
