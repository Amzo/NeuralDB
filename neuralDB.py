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
        self.totalError = 0

    def connect_to_db(self, performance):
        self.connection = sqlite3.connect(self.databaseLocation, isolation_level='DEFERRED')
        self.database = self.connection.cursor()
        if performance:
            self.database.execute('''PRAGMA synchronous = OFF''')
            self.database.execute('''PRAGMA journal_mode = OFF''')

    def create_table(self, tablename=str, tabletype=str, counter=int):
        self.database.execute("DROP TABLE if exists {};".format(tablename))

        if tabletype == "Input":
            sqlCommand = '''CREATE TABLE if not exists inputNeuron (neuronID INTEGER PRIMARY KEY, x FLOAT)'''
        elif tabletype == "Dense":
            sqlCommand = 'CREATE TABLE if not exists Dense{} (neuronID INTEGER PRIMARY KEY, output FLOAT)'.format(
                counter)
        elif tabletype == "Error":
            if tablename == "inputErrors":
                sqlCommand = 'CREATE TABLE if not exists inputErrors (neuronID INTEGER, errors FLOAT, ' \
                             'FOREIGN KEY (NeuronID) References inputNeuron (neuronID));'
            else:
                sqlCommand = 'CREATE TABLE if not exists Error{} (neuronID INTEGER, errors FLOAT, ' \
                             'FOREIGN KEY (NeuronID) References Dense{} (neuronID));'.format(counter, counter)
        elif tabletype == "Connections":
            sqlCommand = 'CREATE TABLE if not exists Connections{} (neuronID INTEGER, connectedNeurons INTEGER, ' \
                         'weights FLOAT, gradient FLOAT, FOREIGN KEY (ConnectedNeurons) References '.format(counter)

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

    def add_table_entry(self, id=int, x=float, tableName=str):
        data = [id, x]

        if tableName == "inputNeuron":
            sqlCommand = '''INSERT INTO inputNeuron (neuronID, x)
                        VALUES( ?,	?)'''
        elif tableName == "inputErrors":
            sqlCommand = '''INSERT INTO inputErrors (neuronID, errors)
                                    VALUES( ?,	?)'''

        self.database.execute(sqlCommand, data)

    def get_input(self, table):
        sqlCommand = 'SELECT COUNT(*) FROM {};'.format(table)
        self.database.execute(sqlCommand)
        inputRows = self.database.fetchone()[0]

        return inputRows

    def dense_table_exists(self):
        sqlCommand = 'SELECT count(*) FROM sqlite_master where type="table" and tbl_name=\"Dense{}\";'.format(
            self.denseCounter)
        self.database.execute(sqlCommand)

        # returns 1 if found 0 if not
        return self.database.fetchone()[0]

    def get_dense_layer(self):
        tableFound = True

        if self.denseCounter > 1:
            self.denseCounter = 0

        while tableFound:
            self.denseCounter += 1
            if not self.dense_table_exists():
                # off by 1
                self.denseCounter -= 1
                tableFound = False

    def addToInputX(self, dataInput):
        for y in range(1, len(dataInput) + 1):
            sqlCommand = "UPDATE inputNeuron SET x = {} WHERE neuronID = {};".format(dataInput[y - 1] / 255, y)
            self.database.execute(sqlCommand)

        self.connection.commit()

    # To save on computational costs, don't fully connect the layers, instead if the size of the previous layer
    # is greater than 20 neurons, only connect 10% of the nodes to each following node.
    def connect_layers(self, neuronID=int, table1=str, table2=str, layer=int, size=int):
        # get previous table number of neurons
        input1 = self.get_input(table1)
        input2 = size
        if input1 < 20:
            for table1Neurons in range(1, input1 + 1):
                weights = random.uniform(-1, 1)

                self.database.executemany(
                    "INSERT INTO Connections{} (neuronID, connectedNeurons, weights, gradient) VALUES (?,?,?,?)".format(
                        layer),
                    [(table1Neurons, neuronID, weights, 0)])
                self.writeCount += 1
            # add a final input for the bias to the connected node.
            weights = random.uniform(-1, 1)
            self.database.executemany(
                "INSERT INTO Connections{} (connectedNeurons, weights, gradient) VALUES (?, ?, ?)".format(layer),
                [(neuronID, weights, 0)])
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
                    "INSERT INTO Connections{} (neuronID, connectedNeurons, weights, gradient) VALUES (?,?,?,?)".format(
                        layer),
                    [(table1Neurons, neuronID, weights, 0)])
                self.writeCount += 1
            weights = random.uniform(-1, 1)
            self.database.executemany(
                "INSERT INTO Connections{} (connectedNeurons, weights, gradient) VALUES (?, ?, ?)".format(layer),
                [(neuronID, weights, 0)])

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

    def update_deltas(self, count, actual, target):
        sqlCommand = "update Error2 set errors = ({} - {}) * {} * (1 - {}) where neuronID = {}".format(actual, target, actual, actual, count)

        self.database.execute(sqlCommand)

    def calculate_gradients(self, layer, label):
        # gradients for the output layer
        # needs optimized
        if layer == self.denseCounter:
            neurons = self.get_input("Dense{}".format(layer))
            prevLayer = layer - 1
            for neuron in range(1, neurons + 1):
                sqlCommand = "UPDATE Connections{} SET gradient = t.gradient " \
                             "FROM " \
                             "( " \
                                "SELECT Connections{}.neuronID as neuron, connectedNeurons, errors * Dense{}.output AS gradient " \
                                "FROM Connections{} " \
                                "JOIN Dense{} on Connections{}.neuronID = Dense{}.neuronID " \
                                "JOIN Error{} on Connections{}.connectedNeurons = Error{}.neuronID " \
                                "WHERE connectedNeurons = {} " \
                             ") t " \
                             "WHERE Connections{}.neuronID = t.neuron " \
                             "AND Connections{}.connectedNeurons = {}".format(layer, layer, prevLayer, layer, prevLayer, layer, prevLayer, layer, layer, layer, neuron, layer, layer, neuron)

                self.database.execute(sqlCommand)

                # bias gradients are just the deltas
                sqlCommand = "update Connections{} set gradient = t.gradient " \
                             "FROM " \
                             "( " \
                                 "SELECT errors as gradient " \
                                 "FROM Error{} " \
                                 "WHERE neuronID = {} " \
                             ") t " \
                             "WHERE Connections{}.neuronID is null " \
                             "AND Connections{}.connectedNeurons = {}".format(layer, layer, neuron, layer, layer, neuron)

                self.database.execute(sqlCommand)
            self.commit_changes()
        else:
            # work through the next layers
            neurons = self.get_input("Dense{}".format(layer))
            for neuron in range(1, neurons + 1):
                sqlCommand = "update Error{} set errors = t.delta " \
                             "from " \
                             "( " \
                                 "select sum(errors * weights) as delta " \
                                 "from Connections{} " \
                                 "join Error{} on Error{}.neuronID = Connections{}.connectedNeurons " \
                                 "where Connections{}.neuronID = {} " \
                             ") t " \
                             "where neuronID = {}".format(layer, layer + 1, layer + 1, layer +1, layer + 1, layer + 1, neuron, neuron)

                self.database.execute(sqlCommand)

            self.commit_changes()
            # part two of the calculations is the same as deltas for ouput nodes. outb1 * (1 - outb1)
            if layer == 1:
                neurons = self.get_input("inputNeuron")
                for neuron in range(1, neurons + 1):
                    sqlCommand = "update Connections{} set gradient = t.result " \
                                 "FROM " \
                                 "( " \
                                 "SELECT Connections{}.connectedNeurons as target, (errors * (output * ( 1 - output)) * x) as result " \
                                 "FROM Connections{} " \
                                 "JOIN Error{} on Connections{}.connectedNeurons = Error{}.neuronID " \
                                 "JOIN inputNeuron on Connections{}.neuronID = inputNeuron.neuronID " \
                                 "JOIN Dense{} on Connections{}.connectedNeurons = Dense{}.neuronID " \
                                 "WHERE Connections{}.neuronID = {} " \
                                 ") t " \
                                 "WHERE Connections{}.connectedNeurons = t.target " \
                                 "AND Connections{}.neuronID = {}".format(layer, layer, layer, layer, layer, layer, layer, layer, layer, layer, layer, neuron, layer, layer, neuron)

                    self.database.execute(sqlCommand)

                # finally update the bais gradients
                sqlCommand = "update Connections{} set gradient = t.result " \
                             "FROM " \
                             "( " \
                                "SELECT Connections{}.connectedNeurons as target, errors * (output * ( 1 - output)) as result " \
                                "FROM Connections{} " \
                                "JOIN Error{} on Connections{}.connectedNeurons = Error{}.neuronID " \
                                "JOIN Dense{} on Connections{}.connectedNeurons = Dense{}.neuronID " \
                                "WHERE Connections{}.neuronID is null " \
                             ") t " \
                             "WHERE Connections{}.connectedNeurons = t.target " \
                             "AND Connections{}.neuronID is null".format(layer, layer, layer, layer, layer, layer, layer, layer, layer, layer, layer, layer)
                self.database.execute(sqlCommand)
                self.commit_changes()

    def update_weights(self, layer):
        # weight - learning rate * gradient
        sqlCommand = "update Connections{} set weights = Connections{}.weights - 0.5 * gradient".format(layer, layer)
        self.database.execute(sqlCommand)
        self.commit_changes()

    def update_errors(self, layer):
        if layer == -1:
            for idx, y in enumerate(self.outputErrors):
                sqlCommand = "UPDATE Error{} set errors = {} where NeuronID = {}".format(2, y, idx + 1)
                self.database.execute(sqlCommand)
        else:
            # first layer is always the input layer, due to naming of layers handle this
            if layer == 1:
                neurons = self.get_input("inputNeuron")
                outputNeurons = self.get_input("Dense{}".format(layer))
            else:
                neurons = self.get_input("Dense{}".format(layer - 1))
                outputNeurons = self.get_input("Dense{}".format(layer))

            for i in range(1, neurons + 1):
                errorResults = []
                for x in range(1, outputNeurons + 1):
                    sqlCommand = "SELECT (weights * t.error) " \
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

    def feed_forward_network(self, layer):
        # SQlite doesn't support Eeulers number, so hard code it as 2.71828 to allow sqlite to do the sigmoid activation.
        if layer == 1:
            sqlCommand = "UPDATE Dense1 SET output = 1 / (1 + pow(2.71828, -(t.sumWeights + ( 1 * w.bias)))) " \
                         "FROM " \
                         "(" \
                         "SELECT Connections1.connectedNeurons, " \
                         "SUM(inputNeuron.x * Connections1.weights) as sumWeights " \
                         "FROM Connections1 JOIN inputNeuron " \
                         "ON Connections1.neuronID = inputNeuron.neuronID " \
                         "WHERE Connections1.neuronID is not null " \
                         "group by Connections1.connectedNeurons " \
                         ") t, " \
                         "( " \
                         "SELECT Connections1.connectedNeurons, Connections1.weights as bias " \
                         "FROM Connections1 " \
                         "where Connections1.neuronID is null " \
                         ") w " \
                         "WHERE Dense1.neuronID = t.connectedNeurons and Dense1.neuronID = w.connectedNeurons"
        else:
            sqlCommand = "UPDATE Dense{} SET output = 1 / (1 + pow(2.71828, -(t.sumWeights + ( 1 * w.bias))))" \
                         "FROM ( " \
                         "SELECT Connections{}.connectedNeurons, " \
                         "SUM(Dense{}.output * Connections{}.weights) as sumWeights " \
                         "FROM Connections{} JOIN Dense{} " \
                         "ON Connections{}.neuronID = Dense{}.neuronID " \
                         "WHERE Connections{}.neuronID is not null " \
                         "group by Connections{}.connectedNeurons " \
                         ") t, " \
                         "( " \
                         "SELECT Connections{}.connectedNeurons, Connections{}.weights as bias " \
                         "FROM Connections{} " \
                         "WHERE Connections{}.neuronID is null " \
                         ") w " \
                         "WHERE Dense{}.neuronID = t.connectedNeurons " \
                         "and Dense{}.neuronID = w.connectedNeurons".format(layer, layer, layer - 1, layer, layer,
                                                                            layer - 1, layer, layer - 1, layer, layer,
                                                                            layer, layer, layer, layer, layer, layer)

        self.database.execute(sqlCommand)

        # add bias to output

    def index_database(self, numOfConnections=int):
        for x in range(1, numOfConnections):
            print("indexing connection table {}".format(x))
            sqlCommand = "CREATE index connects{} ON Connections{} ('connectedNeurons' ASC)".format(x, x)
            self.database.execute(sqlCommand)

        sqlCommand = "CREATE index input ON inputNeuron ('neuronID' ASC)"
        self.database.execute(sqlCommand)
