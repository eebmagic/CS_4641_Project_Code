import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from data_loader import DataSet
 
def plot_predictions(plot_data: list, title: str='Linear Regression: Predicted vs Actual'):
    x_test = plot_data[0]
    y_test = plot_data[1]
    y_preds = plot_data[2]
    r2 = plot_data[3]

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_preds, color='deepskyblue', alpha=0.8, linewidths=0)

    m, b = np.polyfit(y_test, y_preds, 1)
    plt.plot(y_test, (m*y_test + b), color='black')

    axis_max = max([max(y_test), max(y_preds)]) + 0.5
    axis_min = min([min(y_test), min(y_preds)]) - 0.5
    plt.axis([axis_min, axis_max, axis_min, axis_max])

    plt.suptitle(title, size=10)
    plt.title(f'R\u00b2 = {round(r2, 8)}       Slope of Trendline = {round(m, 8)}', size=10)
    plt.xlabel('Actual Normalized Price Per Sq. Ft')
    plt.ylabel('Predicted Normalized Price Per Sq. Ft')
    return

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
plot_data = [x_test, y_test, pred, r2]
plot_predictions(plot_data)
plt.show()
