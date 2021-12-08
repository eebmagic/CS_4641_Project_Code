import numpy as np 
import matplotlib.pyplot as plt

from data_loader import DataSet
from plotting import plot_predictions

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras    

def get_data(split: bool=False):
    ds = DataSet()
    X = ds.census
    Y = ds.zillow_actual
    if split:
        return train_test_split(X, Y, test_size=0.20)
    else:
        return X, Y

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

def evaluate(model: keras.Model, X: list, Y: list) -> float:
    eval_batch_amount = int(len(X) / 10)
    evaluation = model.evaluate(X, Y, eval_batch_amount)
    return evaluation[1]

def plot(Y_test: list, Y_predictions: list, all_data: bool=False) -> None:
	plot_data = {
	    'test': Y_test,
	    'pred': Y_predictions
	}
	plot_predictions(plot_data, title='Neural Network: Predicted vs Actual', zillow_actual=True, all_data=all_data)
	plt.show()

def plot_model_all(model_filename: str) -> None:
    model = keras.models.load_model(model_filename)
    X, Y = get_data()
    Y_predictions = model.predict(X)
    plot(Y.flatten(), Y_predictions.flatten(), all_data=True)

def create_model(model_filename: str) -> tuple[keras.Model, float, list[list], list]:
    X_train, X_test, Y_train, Y_test = get_data(split=True)
    model = gen_model()
    model = train(model, X_train, Y_train)
    mape = evaluate(model, X_test, Y_test)
    return model, eval_mape, X_test, Y_test

def create_accepted_model(model_filename: str, acceptable_eval_mape: float, acceptable_r2: float) -> None:
    accepted = False
    while not accepted:
        model, eval_mape, X_test, Y_test = create_model(model_filename)
        
        if eval_mape < 16:
            Y_predictions = model.predict(X_test) 

            if r2_score(Y_test, Y_predictions) > 0.7:
                plot(Y_test.flatten(), Y_predictions.flatten())
                model.save(model_filename)
                accepted = True


if __name__ == "__main__":
    model_filename = 'new_model.h5'

    create_accepted_model(model_filename, acceptable_eval_mape=16, acceptable_r2=0.7)
    plot_model_all(model_filename)
    
