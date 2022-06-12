import time

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
        self.neuralDB.create_table(tableName="inputNeuron", tableType="Input", counter=0)
        self.neuralDB.create_table(tableName="inputErrors", tableType="Error")

        neuron = shape

        # populate with 0.0s
        # remove me later
        for neuronID in range(1, neuron + 1):
            self.neuralDB.addTableEntry(id=neuronID, x=0.0, tableName="inputNeuron")
            self.neuralDB.addTableEntry(id=neuronID, x=0.0, tableName="inputErrors")

        self.neuralDB.commit_changes()
        self.neuralDB.close()

    def dense(self, neurons=int):
        self.neuralDB.connect_to_db(True)
        self.neuralDB.create_table(tableName="Connections{}".format(self.denseCounter), tableType="Connections",
                                   counter=self.denseCounter)
        self.neuralDB.create_table(tableName="Dense{}".format(self.denseCounter), tableType="Dense",
                                   counter=self.denseCounter)
        self.neuralDB.create_table(tableName="Error{}".format(self.denseCounter), tableType="Error",
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
            self.neuralDB.getWeights(x)

        self.neuralDB.commit_changes()

    def __backpass(self):
        # get error
        errors = self.neuralDB.getResults()
        self.neuralDB.outputErrors = []

        for y in errors:
            # each index represents the corresponding output neuron
            self.neuralDB.outputErrors.append(self.labels[0] - y[0])

        self.neuralDB.update_errors(-1)
        # Need to loop through each layer backwards
        for x in range(self.neuralDB.denseCounter, 0, -1):
            self.neuralDB.update_errors(x)

        self.neuralDB.commit_changes()

        # Calculate the gradients
        for x in range(self.neuralDB.denseCounter, 0, -1):
            self.neuralDB.calculate_gradients(x)

    def fit(self, train=list, labels=list, epochs=int):
        self.labels = labels

        self.neuralDB.connect_to_db(True)
        self.neuralDB.getDenseLayers()
        print("Starting Training")
        start_time = time.time()
        for i in range(1, epochs):
            self.__fowardPass(train)
            self.__backpass()

        print("finished training {} epochs in  {} seconds".format(epochs, time.time() - start_time))

    # optimize the database for faster searching and joins, increases speed substantially
    def optimize(self):
        self.neuralDB.connect_to_db(True)
        self.neuralDB.indexDatabase(self.denseCounter)
        self.neuralDB.commit_changes()
        self.neuralDB.close()
