import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from data_loader import DataSet
from feature_selection import forward_feature_selection

def get_raw_data():
    ds = DataSet()
    x = ds.census
    y = ds.zillow.flatten()
    return x, y

def preprocess_data(x, y):
    #forwardResults = forward_feature_selection(x, y, model_init=SVR, param_grid={})
    # Hard coded because the forward selection takes a bit of time
    forwardResults = np.array([3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 26, 27, 29, 30, 33])
    x = x[:, forwardResults]
    return x, y

def fit_ridge(x, y):
    ridge = Ridge(alpha=1.0)
    ridge.fit(x, y)
    return ridge

def get_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    ridge = fit_ridge(x_train, y_train)
    y_preds = ridge.predict(x_test)
    r2 = r2_score(y_test, y_preds)

    plot_data = [x_test, y_test, y_preds, r2]
    return ridge, r2, plot_data

def get_r2_test_list(num_models: int, preprocess: bool=False) -> list:
    x, y = get_raw_data()
    if preprocess:
        x, y = preprocess_data(x, y)

    r2_test_list = []
    for i in range(num_models):
        _, r2, _ = get_model(x, y)
        r2_test_list.append(r2)

    return r2_test_list

def r2_test(num_models: int=1000):
    r2_test_list_raw = get_r2_test_list(num_models)
    mean_r2_raw = np.mean(r2_test_list_raw)
    std_r2_raw = np.std(r2_test_list_raw)

    r2_test_list_preprocessed = get_r2_test_list(num_models, preprocess=True)
    mean_r2_preprocessed = np.mean(r2_test_list_preprocessed)
    std_r2_preprocessed = np.std(r2_test_list_preprocessed)

    return mean_r2_raw, std_r2_raw, mean_r2_preprocessed, std_r2_preprocessed

def plot_predictions(plot_data: list, title: str='Ridge Regression: Predicted vs Actual'):
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
    plt.title(f'R\u00b2 = {round(r2, 8)} \t  Slope of Trendline = {round(m, 8)}', size=10)
    plt.xlabel('Actual Normalized Price Per Sq. Ft')
    plt.ylabel('Predicted Normalized Price Per Sq. Ft')
    return


if __name__ == "__main__":
    x, y = get_raw_data()
    ridge_raw, r2_raw, plot_data = get_model(x, y)
    print(f"Ridge R2 Score with Raw Data: {round(r2_raw, 8)}")
    plot_predictions(plot_data, 'Ridge Regression with Raw Data: Predicted vs Actual')

    x_preprocessed, y_preprocessed = preprocess_data(x, y)
    ridge_preproccessed, r2_preprocessed, plot_data = get_model(x_preprocessed, y_preprocessed)
    print(f"Ridge R2 Score with Forward Selected Features: {round(r2_preprocessed, 8)}")
    plot_predictions(plot_data, 'Ridge Regression with Forward Selected Features: Predicted vs Actual')

    # print("Running Ridge Regression R2 test...")
    mean_r2_raw, std_r2_raw, mean_r2_preprocessed, std_r2_preprocessed = r2_test()

    print(f"Ridge R2 Score Average with Raw Data: {round(mean_r2_raw, 8)} \n\t  with Standard Deviation: {round(std_r2_raw, 8)}")
    print(f"Ridge R2 Score Average with Forward Selected Features: {round(mean_r2_preprocessed, 8)} \n\t  with Standard Deviation: {round(std_r2_preprocessed, 8)}")

    plt.show()

