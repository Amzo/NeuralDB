import numpy as np

import neuralDB


def Input(shape=tuple, databaseLocation=str):
    """
    :param databaseLocation: str
    :param shape: tuple
    """
    assert type(shape) is tuple, "Input shape should be a tuple"

    assert type(databaseLocation) is str, "Database location should be a string"

    database = neuralDB.NeuralDB()
    database.databaseLocation = databaseLocation

    database.createInputLayer(shape=shape)


def Dense(neurons=int, databaseLocation=str):
    database = neuralDB.NeuralDB()
    database.databaseLocation = databaseLocation

    database.createDenseLayer(shape=neurons)


def Compile(databaseLocation=str):
    database = neuralDB.NeuralDB()
    database.databaseLocation = databaseLocation
    database.connect_layers()
    database.addRandomWeights()


def fit(databaseLocation=str, train=None, label=None):
    database = neuralDB.NeuralDB()
    database.databaseLocation = databaseLocation
    database.fit(train, label)
