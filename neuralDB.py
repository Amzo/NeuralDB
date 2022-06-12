import random
import sqlite3

import numpy as np


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
        self.outputErrors = []

    def connect_to_db(self, performance):
        self.connection = sqlite3.connect(self.databaseLocation, isolation_level='DEFERRED')
        self.database = self.connection.cursor()
        if performance:
            self.database.execute('''PRAGMA synchronous = OFF''')
            self.database.execute('''PRAGMA journal_mode = OFF''')

    def create_table(self, tableName=str, tableType=str, counter=int):
        self.database.execute("DROP TABLE if exists {};".format(tableName))

        if tableType == "Input":
            sqlCommand = '''CREATE TABLE if not exists inputNeuron (neuronID INTEGER PRIMARY KEY, x FLOAT)'''
        elif tableType == "Dense":
            sqlCommand = 'CREATE TABLE if not exists Dense{} (neuronID INTEGER PRIMARY KEY, output FLOAT)'.format(
                counter)
        elif tableType == "Error":
            if tableName == "inputErrors":
                sqlCommand = 'CREATE TABLE if not exists inputErrors (neuronID INTEGER, errors FLOAT, ' \
                             'FOREIGN KEY (NeuronID) References inputNeuron (neuronID));'
            else:
                sqlCommand = 'CREATE TABLE if not exists Error{} (neuronID INTEGER, errors FLOAT, ' \
                             'FOREIGN KEY (NeuronID) References Dense{} (neuronID));'.format(counter, counter)
        elif tableType == "Connections":
            sqlCommand = 'CREATE TABLE if not exists Connections{} (neuronID INTEGER, connectedNeurons INTEGER, ' \
                         'weights FLOAT, FOREIGN KEY (ConnectedNeurons) References '.format(counter)

            sqlCommand += 'Dense{} (neuronID), '.format(counter)

            if counter == 1:
                sqlCommand += 'FOREIGN KEY (neuronID) REFERENCES inputNeuron(neuronID));'
            else:
                sqlCommand += 'FOREIGN KEY (neuronID) REFERENCES Dense{} (neuronID));'.format(counter - 1)

        self.database.execute(sqlCommand)

    def commit_changes(self):
        self.connection.commit()

    def close(self):
        self.database.close()
        self.connection.close()

    def addTableEntry(self, id=int, x=float, tableName=str):
        data = [id, x]

        if tableName == "inputNeuron":
            sqlCommand = '''INSERT INTO inputNeuron (neuronID, x)
                        VALUES( ?,	?)'''
        elif tableName == "inputErrors":
            sqlCommand = '''INSERT INTO inputErrors (neuronID, errors)
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
        self.create_table("Connections")
        self.connection.commit()

    def addToInputX(self, dataInput):
        for y in range(1, len(dataInput) + 1):
            sqlCommand = "UPDATE inputNeuron SET x = {} WHERE neuronID = {};".format(dataInput[y - 1], y)
            self.database.execute(sqlCommand)
        self.connection.commit()

    # To save on computational costs, don't fully connect the layers, instead if the size of the previous layer
    # is greater than 20 neurons, only connect 10% of the nodes to each following node.
    def connect_layers(self, neuronID=int, table1=str, table2=str, layer=int, size=int):
        # get previous table number of neurons
        input1 = self.getInput(table1)
        input2 = size
        if input1 < 20:
            for table1Neurons in range(1, input1 + 1):
                weights = random.uniform(-1, 1)

                self.database.executemany(
                    "INSERT INTO Connections{} (neuronID, connectedNeurons, weights) VALUES (?,?, ?)".format(layer),
                    [(table1Neurons, neuronID, weights)])
                self.writeCount += 1
        else:
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

            # print(lowerRange + 1, upperRange, neuronID)
            for table1Neurons in range(lowerRange + 1, upperRange + 1):
                weights = random.uniform(-1, 1)

                self.database.executemany(
                    "INSERT INTO Connections{} (neuronID, connectedNeurons, weights) VALUES (?,?, ?)".format(layer),
                    [(table1Neurons, neuronID, weights)])
                self.writeCount += 1

            self.multiplier += 1

    def getResults(self):
        sqlCommand = "SELECT Dense{}.output " \
                     "FROM Dense{}".format(self.denseCounter, self.denseCounter)
        self.database.execute(sqlCommand)

        return self.database.fetchall()

    def addNeurons(self, neurons, counter=int):
        sqlCommand = 'INSERT INTO Dense{}'.format(counter)
        self.multiplier = 1
        for connectedNeuron in range(1, neurons + 1):
            self.database.execute('INSERT INTO Error{} (NeuronID) VALUES (?)'.format(counter), [connectedNeuron])
            self.database.execute(sqlCommand + " (neuronID) VALUES (?)",
                                  [connectedNeuron])
            if counter == 1:
                self.connect_layers(neuronID=connectedNeuron, table1="inputNeuron",
                                    table2="Dense{}".format(counter), layer=counter, size=neurons)
            else:
                self.connect_layers(neuronID=connectedNeuron, table1="Dense{}".format(counter - 1),
                                    table2="Dense{}".format(counter), layer=counter, size=neurons)

    def calculate_gradients(self, layer):
        #  1) for the gradient first get all the output derivative of leaky ReLu for the current layer
        #  2) multiply that output derivative by the output error
        #  3) times that result by the learning rate, using a constant 0.001 for now

        # derivative of output, multiply it by the errors, then times by learning rate.
        # use sql for all calculations
        sqlCommand = "SELECT 0.001 * errors * t.results " \
                     "FROM Dense{}, Error{}, " \
                     "( " \
                     "SELECT IIF(output>0, 1, 0.01) as results " \
                     "FROM Dense{}" \
                     ") t " \
                     "GROUP by Dense{}.neuronID".format(layer, layer, layer, layer)

        self.database.execute(sqlCommand)

        weight_updates = self.database.fetchall()

        numNeurons = self.getInput("Dense{}".format(layer))
        # time all weights connected to ouput 1 by gradient
        for x in range(1, numNeurons + 1):
            sqlCommand = "Select {} * weights from Connections{} " \
                         "where connectedNeurons = {}".format(str(weight_updates[x - 1][0]), layer, x)

            self.database.execute(sqlCommand)
            results = self.database.fetchall()
            weight_deltas = []
            for idx, val in enumerate(results):
                # multiply each weight with the gradient to get the deltas
                sqlCommand = "SELECT weights * {} " \
                             "FROM Connections{} " \
                             "WHERE NeuronID = {} and connectedNeurons = {}".format(val[0], layer, idx + 1, x)

                self.database.execute(sqlCommand)

                weight_deltas.append(self.database.fetchone())

            # update the current weight by adding the  delta
            for idx, y in enumerate(weight_deltas):
                sqlCommand = "UPDATE Connections{} set weights = weights + {} " \
                             "WHERE neuronID  = {} and ConnectedNeurons = {}".format(layer, y[0], idx + 1, x)

                self.database.execute(sqlCommand)

        self.commit_changes()

        # multiply the gradients by the output to compute the deltas

    def update_errors(self, layer):
        if layer == -1:
            for idx, y in enumerate(self.outputErrors):
                sqlCommand = "UPDATE Error{} set errors = {} where NeuronID = {}".format(2, y, idx + 1)
                self.database.execute(sqlCommand)
        else:
            # first layer is always the input layer, due to naming of layers handle this
            if layer == 1:
                neurons = self.getInput("inputNeuron")
                outputNeurons = self.getInput("Dense{}".format(layer))
            else:
                neurons = self.getInput("Dense{}".format(layer - 1))
                outputNeurons = self.getInput("Dense{}".format(layer))

            for i in range(1, neurons + 1):
                errorResults = []
                for x in range(1, outputNeurons + 1):
                    sqlCommand = "SELECT weights * t.error " \
                                 "FROM Connections{}, " \
                                 "( " \
                                 "SELECT Error{}.errors as error " \
                                 "FROM Error{} " \
                                 "WHERE Error{}.neuronID = {} " \
                                 ") t " \
                                 "WHERE neuronID = {} AND connectedNeurons = {}".format(layer, layer, layer, layer,
                                                                                        x, i, x)
                    self.database.execute(sqlCommand)
                    errorResults.append(self.database.fetchone())

                sum = np.sum(errorResults)

                if (layer - 1) == 0:
                    sqlCommand = "UPDATE inputErrors set errors = {} where NeuronID = {}".format(sum, i)
                else:
                    sqlCommand = "UPDATE Error{} set errors = {} where NeuronID = {}".format(layer - 1, sum, i)
                self.database.execute(sqlCommand)

    def getWeights(self, layer):
        # Leaky ReLU where A is 0.01 and Z is the input * weights
        if layer == 1:
            sqlCommand = "UPDATE Dense1 SET output = MAX(0.01 * t.sumWeights, t.sumWeights) " \
                         "FROM " \
                         "(" \
                         "SELECT Connections1.connectedNeurons, " \
                         "SUM(inputNeuron.x * Connections1.weights) as sumWeights " \
                         "FROM Connections1 JOIN inputNeuron " \
                         "ON Connections1.neuronID = inputNeuron.neuronID " \
                         "group by Connections1.connectedNeurons " \
                         ") t " \
                         "WHERE Dense1.neuronID = t.connectedNeurons"
        else:
            sqlCommand = "UPDATE Dense{} SET output = MAX(0.01 * t.sumWeights, t.sumWeights) " \
                         "FROM " \
                         "(" \
                         "SELECT Connections{}.connectedNeurons, " \
                         "SUM(Dense{}.output * Connections{}.weights) as sumWeights " \
                         "FROM Connections{} JOIN Dense{} " \
                         "ON Connections{}.neuronID = Dense{}.neuronID " \
                         "group by Connections{}.connectedNeurons " \
                         ") t " \
                         "WHERE Dense{}.neuronID = t.connectedNeurons".format(layer, layer, layer - 1, layer, layer,
                                                                              layer - 1, layer, layer - 1, layer, layer)

        self.database.execute(sqlCommand)

    def indexDatabase(self, numOfConnections=int):
        for x in range(1, numOfConnections):
            print("indexing connection table {}".format(x))
            sqlCommand = "CREATE index connects{} ON Connections{} ('connectedNeurons' ASC)".format(x, x)
            self.database.execute(sqlCommand)
