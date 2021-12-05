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
print(f'\nPredition: {pred}')

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
plt.show()
