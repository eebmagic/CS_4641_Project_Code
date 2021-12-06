import numpy as np
from sklearn.linear_model import Lasso
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

# Try multiple alpha values
alphas = [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
for alpha in alphas:
    print(f'\nalpha: {alpha}')

    # Fit model
    reg = Lasso(alpha=alpha).fit(x_train, y_train)

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
    title = f'Lasso Regression (alpha={alpha})'
    filename = f'lasso_alpha_{alpha}'
    plot_predictions(plot_data, title=title, filename=filename)

plt.show()
