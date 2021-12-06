import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_predictions(plot_data: dict,
    title: str='DEFAULT TITLE: Predicted vs Actual', filename: str=None, zillow_actual: bool=False):
    '''
    Parameters:
        plot_data: A dictionary with 'test' (the true price target),
            and 'pred' (the prices predicted by model).

        title: Title to be used for the plot.

        filename: Name to be used for image file export. If None, then no file will save.
    '''
    y_test = plot_data['test']
    y_preds = plot_data['pred']

    dists = np.abs(y_test - y_preds)
    MIN, MAX = dists.min(), dists.max()
    m = interp1d([MIN, MAX], [0,1])
    dists = m(dists)

    r2 = r2_score(y_test, y_preds)

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_preds, c=dists, cmap=plt.get_cmap('plasma'))

    m, b = np.polyfit(y_test, y_preds, 1)
    plt.plot(y_test, y_test, color='gray', linestyle='dotted', linewidth=1)
    plt.plot(y_test, (m*y_test + b), color='black')

    axis_max = max([max(y_test), max(y_preds)]) * 1.2
    axis_min = min([min(y_test), min(y_preds), 0]) * 1.2
    plt.axis([axis_min, axis_max, axis_min, axis_max])

    plt.suptitle(title, size=10)
    plt.title(f'R\u00b2 = {round(r2, 8)}       Slope of Trendline = {round(m, 8)}', size=10)

    if zillow_actual:
        plt.xlabel('Actual Zillow Rent Price Index')
        plt.ylabel('Predicted Zillow Rent Price Index')
    else:
        plt.xlabel('Actual Normalized Price Per Sq. Ft')
        plt.ylabel('Predicted Normalized Price Per Sq. Ft')

    if filename:
        plt.savefig(f'image_exports/{filename}.png', bbox_inches='tight')

    return
