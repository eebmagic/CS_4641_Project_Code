import numpy as np 
import matplotlib.pyplot as plt

from data_loader import DataSet
from plotting import plot_predictions

from sklearn.model_selection import train_test_split
from tensorflow import keras

def get_train_test_data() -> tuple[list[list], list[list], list, list]:
    ds = DataSet()
    X = ds.census
    Y = ds.zillow_actual
    return train_test_split(X, Y, test_size=0.20)

def gen_model() -> keras.Model:
    visible = keras.Input(shape=(34, ))
    hidden1 = keras.layers.Dense(200, activation='relu')(visible)
    hidden2 = keras.layers.Dense(200, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(200, activation='relu')(hidden2)
    hidden4 = keras.layers.Dense(200, activation='relu')(hidden3)
    hidden5 = keras.layers.Dense(200, activation='relu')(hidden4)
    hidden6 = keras.layers.Dense(200, activation='relu')(hidden5)
    hidden7 = keras.layers.Dense(200, activation='relu')(hidden6)
    hidden8 = keras.layers.Dense(200, activation='relu')(hidden7)
    output = keras.layers.Dense(1, activation='relu')(hidden8)
    model = keras.Model(inputs=visible, outputs=output)
    print(model.summary())

    model.compile(optimizer='adam', loss='mse', metrics='mape')
    return model

def train(model: keras.Model, X: list, Y: list) -> keras.Model:
    batch_amount = int(len(X) / 10)
    epoch_amount = 50

    model.fit(X, Y, batch_size=batch_amount, epochs=epoch_amount, validation_split=0.1)
    return model

def evaluate(model: keras.Model, X: list, Y: list) -> None:
    eval_batch_amount = int(len(X) / 10)
    model.evaluate(X, Y, eval_batch_amount)

def plot(Y_test: list, Y_predictions: list) -> None:
	plot_data = {
	    'test': Y_test,
	    'pred': Y_predictions
	}
	plot_predictions(plot_data, title='Neural Network: Predicted vs Actual', zillow_actual=True)
	plt.show()


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = get_train_test_data()

    model = gen_model()
    model = train(model, X_train, Y_train)

    evaluate(model, X_test, Y_test)
    Y_predictions = model.predict(X_test)
    plot(Y_test.flatten(), Y_predictions.flatten())

    model.save('model.h5')
    
