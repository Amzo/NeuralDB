import time

import numpy as np
import tensorflow.keras as tk
from keras.utils import to_categorical
import Model
import cProfile

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tk.datasets.cifar10.load_data()

    model = Model.Model(database="./neuralModel.db")
    model.input(shape=1024)
    start_time = time.time()
    model.dense(1024)
    model.dense(10)

    #print("Wrote {} connections between {} layers in {} at {} connections per second".format(
    #    model.neuralDB.writeCount, model.denseCounter, time.time() - start_time,
    #    round(model.neuralDB.writeCount / (time.time() - start_time))))

    model.optimize()
    #xTrain = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    #yTrain = np.array([[0], [1], [1], [0]])
    encoded = to_categorical(y_train)
    cProfile.run('model.fit(train=x_train, labels=encoded, epochs=1)')

