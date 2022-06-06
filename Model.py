import concurrent
import random
import time

import numpy as np

import activations
import neuralDB
import Errors
import ray
import sys


#@ray.remote
def calculateWeights(count=int, table=str, database=object):
    DB = neuralDB.NeuralDB(database)
    DB.connectToDB(True)
    count, sumResults = DB.getWeights(count, table)
    returnString = []
    #print(sumResults)
    for x in iter(sumResults):
        active = activations.sigmoid(x[0])
        returnString.append("UPDATE dense{} SET output = {} WHERE neuronID = {};".format(table, active, count))
        count += 1
    DB.close()
    return ''.join(returnString)
    # active = activations.sigmoid(sumResults[0])
    # return "UPDATE dense" + str(table) + " SET output = " + str(active) + " WHERE neuronID = " + str(neuron) + ";"
    # database.database.execute("UPDATE dense" + str(table) + " SET output = " + str(active) + " WHERE neuronID = " + str(neuron) + ";")


class Model:
    def __init__(self, database=str, GPU=bool):
        self.neuralDB = neuralDB.NeuralDB(database)
        self.debug = False
        self.denseCounter = 1
        self.gpu = GPU

    # take an input tuple and create a table for all the inputs
    def input(self, shape=tuple):
        """
        :param shape: tuple
        """
        # if type(shape) is not tuple:
        #    raise Errors.InvalidInputShape("Inputshape should be a tuple")

        self.neuralDB.connectToDB(True)
        self.neuralDB.createTable(tableName="inputNeuron", tableType="Input", counter=0)

        # (1, 1) pixel image needs to neurons, handle that as shape[0] * shape[1] would be 1 if both values are 1
        # if shape[0] == 1 and shape[1] == 1:
        #    neuron = 2
        # else:
        #    neuron = shape[0] * shape[1]

        neuron = shape
        if self.debug:
            print("Populating {} neurons in table inputNeuron".format(neuron))

        # populate with 0.0s
        # remove me later
        for neuronID in range(1, neuron + 1):
            self.neuralDB.addTableEntry(id=neuronID, x=0.0, tableName="inputNeuron")

        self.neuralDB.commitChanges()
        self.neuralDB.close()

    def dense(self, neurons=int):
        self.neuralDB.connectToDB(True)
        self.neuralDB.createTable(tableName="Connections{}".format(self.denseCounter), tableType="Connections",
                                  counter=self.denseCounter)
        self.neuralDB.createTable(tableName="Dense{}".format(self.denseCounter), tableType="Dense",
                                  counter=self.denseCounter)

        if self.debug:
            print("Populating {} neurons in table dense{}".format(neurons, self.denseCounter))

        self.neuralDB.addNeurons(neurons, counter=self.denseCounter)

        self.neuralDB.commitChanges()
        self.neuralDB.close()
        # every time dense counter is called, increment our dense layer count
        self.denseCounter += 1

    def sumWeights(self, table=int, weights=list, neuron=int):
        print(weights)
        sumResults = np.sum(weights)
        print(sumResults)
        self.neuralDB.updateOutputNeuron(table=table, neuron=neuron, sum=sumResults)

    def calculateWeights(self, neuron=int, table=str):
        sumResults = self.neuralDB.getWeights(neuron, table)
        self.neuralDB.updateOutputNeuron(table=table, neuron=neuron, sum=sumResults[0])

    def __fowardPass(self, dataInput=list):
        self.neuralDB.addToInputX(dataInput=dataInput)

        # loop through the layers to perform updates
        # each update to each neuron is done simultaneously as they don't depend on each other.
        # ray allows executing them in parallel. speed is down from 24s to 5s on a complex model.
        start_time = time.time()

        for x in range(1, self.neuralDB.denseCounter + 1):
            numOfNeurons = self.neuralDB.getInput("Dense{}".format(x))
            #result_ids = [calculateWeights.remote(y, x, self.neuralDB.databaseLocation) for y in
            #              range(1, 17)]  # numOfNeurons + 1)]
            results = []
            y=1
            results.append(self.neuralDB.calculateWeights(y, x))

            self.neuralDB.database.executescript(''.join(results))
            # updates the weights before doing the next layer
            # this seems to be the fastest way. join method 5.6s, map 5.4 seconds, list comprehension 5.3s and the loop takes 5.2s
            #for j in ray.get(result_ids):
            #self.neuralDB.database.executescript(j)
            #del result_ids
        print("forward pass finished in {} seconds".format(time.time() - start_time))
        self.neuralDB.commitChanges()

    def fit(self, train=list, labels=list):
        #if not ray.is_initialized():
        #    ray.init(num_cpus=16, include_dashboard=False)

        self.neuralDB.connectToDB(True)
        self.neuralDB.getDenseLayers()
        print("Starting Forward Pass")
        self.__fowardPass(train)


    # optimize the database for faster searching and joins, increases speed substantially
    def optimize(self):
        self.neuralDB.connectToDB(True)
        self.neuralDB.indexDatabase(self.denseCounter)
        self.neuralDB.commitChanges()
        self.neuralDB.close()
