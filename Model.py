import time
from random import randint

import numpy as np
from tqdm import tqdm

import neuralDB


class Model:
    def __init__(self, database=str):
        self.labels = None
        self.neuralDB = neuralDB.NeuralDB(database)
        self.denseCounter = 1

    # take an input tuple and create a table for all the inputs
    def input(self, shape=tuple):
        """
        :param shape: tuple
        """
        # if type(shape) is not tuple:
        #    raise Errors.InvalidInputShape("Inputshape should be a tuple")

        self.neuralDB.connect_to_db(True)
        self.neuralDB.create_table(tablename="inputNeuron", tabletype="Input", counter=0)
        self.neuralDB.create_table(tablename="inputErrors", tabletype="Error")

        neuron = shape

        # populate with 0.0s
        # remove me later
        for neuronID in range(1, neuron + 1):
            self.neuralDB.add_table_entry(id=neuronID, x=0.0, tableName="inputNeuron")
            self.neuralDB.add_table_entry(id=neuronID, x=0.0, tableName="inputErrors")

        self.neuralDB.commit_changes()
        self.neuralDB.close()

    def dense(self, neurons=int):
        self.neuralDB.connect_to_db(True)
        self.neuralDB.create_table(tablename="Connections{}".format(self.denseCounter), tabletype="Connections",
                                   counter=self.denseCounter)
        self.neuralDB.create_table(tablename="Dense{}".format(self.denseCounter), tabletype="Dense",
                                   counter=self.denseCounter)
        self.neuralDB.create_table(tablename="Error{}".format(self.denseCounter), tabletype="Error",
                                   counter=self.denseCounter)

        self.neuralDB.addNeurons(neurons, counter=self.denseCounter)

        self.neuralDB.commit_changes()
        self.neuralDB.close()
        # every time dense counter is called, increment our dense layer count
        self.denseCounter += 1

    def __fowardPass(self, dataInput=list):
        self.neuralDB.addToInputX(dataInput=dataInput)

        # loop through the layers to perform updates
        # each update to each neuron is done simultaneously as they don't depend on each other.
        # ray allows executing them in parallel. speed is down from 24s to 5s on a complex model.

        for x in range(1, self.neuralDB.denseCounter + 1):
            self.neuralDB.feed_forward_network(x)

        self.neuralDB.commit_changes()

    def __backpass(self, label):
        # get error
        errors = self.neuralDB.getResults()
        results = 0
        self.neuralDB.outputErrors = []

        # tidy this stuff up
        count = 0
        for y in errors:
            # each index represents the corresponding output neuron
            # (actual – forecast)2
            self.neuralDB.outputErrors.append((label[count] - y[0]) ** 2)

            # deltas are (actual - target) * actual * ( 1 - actual)
            self.neuralDB.update_deltas(count + 1, y[0], label[count])
            count += 1

        self.neuralDB.commit_changes()

        # Mean Squared error  (1/n) * Σ(actual – forecast)2
        self.neuralDB.totalError = 1 / 10 * sum(self.neuralDB.outputErrors)

        # Need to loop through each layer backwards
        for layer in range(self.neuralDB.denseCounter, 0, -1):
            self.neuralDB.calculate_gradients(layer, label)

        # updates the weights
        for layer in range(1, self.neuralDB.denseCounter + 1):
            self.neuralDB.update_weights(layer)

    def fit(self, train=list, labels=list, epochs=int):
        self.labels = labels

        self.neuralDB.connect_to_db(True)
        self.neuralDB.get_dense_layer()

        print("Starting Training")
        start_time = time.time()
        #pbar = tqdm(range(len(train)))
        for i in range(1, epochs + 1):
            print("Epoch: {}".format(i))
            #for x in pbar:
            idx = randint(0, len(train) - 1)
            self.__fowardPass(train[idx][:, :, 0].flatten())
            self.__backpass(labels[idx])
            #pbar.set_description("Current Loss {}".format(self.neuralDB.totalError, '.8f'))

        print("finished training {} epochs in  {} seconds".format(epochs, time.time() - start_time))

    def predict(self, train=list):
        self.neuralDB.connect_to_db(True)
        self.neuralDB.get_dense_layer()
        self.__fowardPass(train)
        found = self.neuralDB.getResults()
        print(found[0])

    # optimize the database for faster searching and joins, increases speed substantially
    def optimize(self):
        self.neuralDB.connect_to_db(True)
        self.neuralDB.index_database(self.denseCounter)
        self.neuralDB.commit_changes()
        self.neuralDB.close()
