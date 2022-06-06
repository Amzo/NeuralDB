import itertools
import sqlite3
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor

import cupy
import numpy as np

import activations


class NeuralDB:
    def __init__(self, database=str):
        self.multiplier = int
        self.results = None
        self.y = None
        self.x = None
        self.inputRows = int
        self.database = None
        self.connection = None
        self.databaseLocation = database
        self.denseCounter = 1
        self.denseExists = False
        self.TIMEOUT = 20
        self.sum = 0.0
        self.writeCount = 0

    def connectToDB(self, performance):
        self.connection = sqlite3.connect(self.databaseLocation, isolation_level='DEFERRED')
        self.database = self.connection.cursor()
        if performance:
            self.database.execute('''PRAGMA synchronous = OFF''')
            self.database.execute('''PRAGMA journal_mode = OFF''')

    def createTable(self, tableName=str, tableType=str, counter=int):
        self.database.execute("DROP TABLE if exists {};".format(tableName))

        if tableType == "Input":
            sqlCommand = '''CREATE TABLE if not exists inputNeuron (neuronID INTEGER PRIMARY KEY, x FLOAT)'''
        elif tableType == "Dense":
            sqlCommand = 'CREATE TABLE if not exists Dense{} (neuronID INTEGER PRIMARY KEY, output FLOAT)'.format(
                counter)
        elif tableType == "Connections":
            sqlCommand = 'CREATE TABLE if not exists Connections{} (neuronID INTEGER, connectedNeurons INTEGER, weights FLOAT, FOREIGN KEY (ConnectedNeurons) References '.format(
                counter)

            sqlCommand += 'Dense{} (neuronID), '.format(counter)

            if counter == 1:
                sqlCommand += 'FOREIGN KEY (neuronID) REFERENCES inputNeuron(neuronID));'
            else:
                sqlCommand += 'FOREIGN KEY (neuronID) REFERENCES Dense{} (neuronID));'.format(counter - 1)

        self.database.execute(sqlCommand)

    def commitChanges(self):
        self.connection.commit()

    def close(self):
        self.database.close()
        self.connection.close()

    def addTableEntry(self, id=int, x=float, tableName=str):
        data = [id, x]

        if tableName == "inputNeuron":
            sqlCommand = '''INSERT INTO inputNeuron (neuronID, x)
                        VALUES( ?,	?)'''

        self.database.execute(sqlCommand, data)

    def getInput(self, table):
        sqlCommand = 'SELECT COUNT(*) FROM {};'.format(table)
        self.database.execute(sqlCommand)
        inputRows = self.database.fetchone()[0]

        return inputRows

    def denseTableExists(self):
        sqlCommand = 'SELECT count(*) FROM sqlite_master where type="table" and tbl_name=\"Dense{}\";'.format(
            self.denseCounter)
        self.database.execute(sqlCommand)

        # returns 1 if found 0 if not
        return self.database.fetchone()[0]

    def getDenseLayers(self):
        tableFound = True

        if self.denseCounter > 1:
            self.denseCounter = 0

        while tableFound:
            self.denseCounter += 1
            if not self.denseTableExists():
                # off by 1
                self.denseCounter -= 1
                tableFound = False

    def addNewKeyToConnections(self):
        self.createTable("Connections")
        self.connection.commit()

    def addToInputX(self, dataInput):
        for y in range(1, len(dataInput) + 1):
            sqlCommand = "UPDATE inputNeuron SET x = {} WHERE neuronID = {};".format(dataInput[y - 1], y)
            self.database.execute(sqlCommand)
        self.connection.commit()

    def updateOutputNeuron(self, table=int, neuron=int, sum=float):
        active = activations.sigmoid(sum)
        self.database.execute(
            "UPDATE Dense{} SET output = {} WHERE neuronID = {};".format(table, active, neuron))

    def connectLayers(self, neuronID=int, table1=str, table2=str, layer=int, size=int):
        # get previous table number of neurons
        input1 = self.getInput(table1)
        input2 = size

        rangeScope = input1 // 10 + (input1 % 10 > 0)

        if self.multiplier == 11:
            self.multiplier = 1
        elif self.multiplier == 10 and rangeScope * self.multiplier > input1:
            upperRange = input1
        elif rangeScope * self.multiplier > input1:
            self.multiplier = 1

        if 'upperRange' not in vars():
            upperRange = rangeScope * self.multiplier

        lowerRange = upperRange - rangeScope

        #print(lowerRange + 1, upperRange, neuronID)
        for table1Neurons in range(lowerRange + 1, upperRange + 1):
            weights = random.uniform(-1, 1)

            self.database.executemany(
                "INSERT INTO Connections{} (neuronID, connectedNeurons, weights) VALUES (?,?, ?)".format(layer),
                [(table1Neurons, neuronID, weights)])
            self.writeCount += 1

        self.multiplier += 1

    def calculateWeights(self, count=int, table=str):
        count, sumResults = self.getWeights(count, table)
        returnString = []

        for x in iter(sumResults):
            active = activations.sigmoid(x[0])
            returnString.append("UPDATE dense{} SET output = {} WHERE neuronID = {};".format(table, active, count))
            count += 1
        return ''.join(returnString)

    def addNeurons(self, neurons, counter=int):
        sqlCommand = 'INSERT INTO Dense{}'.format(counter)
        self.multiplier = 1
        for connectedNeuron in range(1, neurons + 1):
            self.database.execute(sqlCommand + " (neuronID) VALUES (?)",
                                  [connectedNeuron])
            if counter == 1:
                self.connectLayers(neuronID=connectedNeuron, table1="inputNeuron",
                                   table2="Dense{}".format(counter), layer=counter, size=neurons)
            else:
                self.connectLayers(neuronID=connectedNeuron, table1="Dense{}".format(counter - 1),
                                   table2="Dense{}".format(counter), layer=counter, size=neurons)

    def getWeights(self, count, layer):
        rangeScope = int(self.getInput("Dense{}".format(layer)) / 16)
        upperRange = rangeScope * count
        lowerRange = upperRange - rangeScope
        # print(lowerRange + 1, upperRange)

        if layer == 1:
            sqlCommand = "SELECT SUM(inputNeuron.x * Connections1.weights) " \
                         "FROM Connections1 " \
                         "JOIN inputNeuron " \
                         "ON Connections1.neuronID = inputNeuron.neuronID " \
                         "group by Connections1.connectedNeurons;"
            # "WHERE Connections1.connectedNeurons >= {} and Connections1.connectedNeurons <= {} " \
            # "group by Connections1.connectedNeurons;"#.format(lowerRange, upperRange)
        else:
            sqlCommand = "SELECT SUM(Dense{}.output * Connections{}.weights) " \
                         "FROM Connections{} " \
                         "JOIN Dense{} " \
                         "ON Connections{}.neuronID = Dense{}.neuronID " \
                         "group by Connections{}.connectedNeurons;".format(layer - 1, layer, layer, layer - 1, layer,
                                                                           layer - 1, layer)
            # "WHERE Connections{}.connectedNeurons >= {} and Connections{}.connectedNeurons <= {} " \
            # "group by Connections{}.connectedNeurons;".format(layer - 1, layer, layer, layer - 1, layer,
            # layer - 1, layer, lowerRange, layer, upperRange,
            # layer)

        self.database.execute(sqlCommand)

        return lowerRange + 1, self.database.fetchall()

    def indexDatabase(self, numOfConnections=int):
        for x in range(1, numOfConnections):
            print("indexing connection table {}".format(x))
            sqlCommand = "CREATE index connects{} ON Connections{} ('connectedNeurons' ASC)".format(x, x)
            self.database.execute(sqlCommand)
