import time
from os.path import exists

import Model
import tensorflow as tf
import keras
import numpy as np

if __name__ == '__main__':
    #(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    #model = Model.Model(database="./neuralModel.db", GPU=False)

    #print("Creating Model")
    #if not exists(model.neuralDB.databaseLocation):
    #    model.input(shape=(128, 128))
    #    start_time = time.time()
    #    model.dense(128)
    #    model.dense(64)
    #    model.dense(64)
    #    model.dense(64)
    #    model.dense(64)
    #    model.dense(64)#

    #    print("Wrote {} connections between {} layers in {} at {} connections per second".format(
    #        model.neuralDB.writeCount, model.denseCounter + 1, time.time() - start_time, round(model.neuralDB.writeCount / (time.time() - start_time))))

    #model.optimize()
    #model.fit(train=[0.1, 0.2, 0.3, 0.4, 0.5, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.123, 0.154, 0.85, 0.254, 0.345] * 8 * 8 * 16, labels=[0])

    #print("performing benchmark of forward pass")

    model = Model.Model(database="./neuralModel.db")
    model.input(shape=1*1)
    #start_time = time.time()
    model.dense(2)
    model.dense(1)

    #print("Wrote {} connections between {} layers in {} at {} connections per second".format(
    #     model.neuralDB.writeCount, model.denseCounter, time.time() - start_time, round(model.neuralDB.writeCount / (time.time() - start_time))))

    model.optimize()
    model.fit(train=[1, 0.1], labels=[1], epochs=5000)
