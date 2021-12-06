from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


from data_loader import DataSet
from plotting import plot_predictions

ds = DataSet()
x = ds.census
y = ds.zillow.flatten()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]
#svr = GridSearchCV(svm.SVR(epsilon = 0.01), parameters, cv = 10)
regr = svm.SVR(kernel="rbf", C=10000, gamma=0.0001).fit(x_train, y_train)

y_pred = regr.predict(x_test)
plot_data = {
    "test": y_test,
    "pred": y_pred
}

plot_predictions(plot_data, title='SVR: Predicted vs Actual', filename='SVR')
plt.show()