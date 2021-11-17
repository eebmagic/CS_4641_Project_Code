from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_loader import DataSet
from feature_selection import forward_feature_selection
import numpy as np 

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

if __name__ == "__main__":
    x, y = get_raw_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05 , random_state=1)
    ridge = fit_ridge(x_train, y_train)
    preds = ridge.predict(x_test)
    print("Ridge R2 Score with Raw Data: {}".format(r2_score(y_test, preds)))


    processed_x, processed_y = preprocess_data(x, y)
    xp_train, xp_test, yp_train, yp_test = train_test_split(processed_x, processed_y, test_size=0.05 , random_state=1)
    ridge2 = fit_ridge(xp_train, yp_train)
    preds2 = ridge2.predict(xp_test)
    print("Ridge R2 Score with Forward Selected Features: {}".format(r2_score(yp_test, preds2)))