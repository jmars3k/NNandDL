import random
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


import mnist_loader
import augmentData
import convNet

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

class Network2(object):

    def __init__(self, sizes, cost = CrossEntropyCost, cosineSimilarity = False,
                 miniBatchSize = 10, eta = 0.5, lmbda = 0.0, dropout = 0, priority = "trainingCost",
                 training_data = None, eval_data= None, test_data = None):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.cosineSimilarity = cosineSimilarity
        if not self.cosineSimilarity:
            self.default_weight_initializer()
        else:
            self.similarity_weight_initializer()
        self.miniBatchSize = miniBatchSize
        self.eta = eta
        self.lmbda = lmbda
        self.dropout = dropout
        self.priority = priority
        self.trainingDataList = list(training_data)
        self.evalDataList = list(eval_data)
        self.testDataList = list(test_data)


    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.oldBiases = self.biases.copy()

        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.oldWeights = self.weights.copy()

    def similarity_weight_initializer(self):
        """

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        similarityRows = self.sizes[1]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.biases[0] = np.zeros((similarityRows,1))
        self.oldBiases = self.biases.copy()

        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.weights[0] = buildSimilarityWeights()
        self.oldWeights = self.weights.copy()

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def initialAugmentation(self):
        self.SGD(1)     # run 1 epoch of SGD
        accuracy = self.accuracy(training = True, evaluation = False, test = False)
        print("Accuracy after 1 epoch on training data: {0} / {1}".format(accuracy, len(self.trainingDataList)))
        self.trainingDataList = augmentData.augmentTrainingData2(self.trainingDataList, self.evalDataList,
                                                                 self.feedforward, initial = True)
        print("The training data now has {} trials".format(len(self.trainingDataList)))
        if not self.cosineSimilarity:
            self.default_weight_initializer()
        else:
            self.similarity_weight_initializer()


    def feedforward(self, a, useDropout = True):
        """Return the output of the network if ``a`` is input."""
#        a = fVect(a)   no improvement, in fact sometimes the network diverges
        thisLayer = 0
        lastLayer = (self.num_layers - 1) -1    #don't include final layer in dropout
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            if useDropout and (thisLayer < lastLayer):
                a *= np.random.binomial([np.ones(a.shape)], 1 - self.dropout)[0] * (1.0 / (1 - self.dropout))
                thisLayer += 1
        return a

    def SGD(self, epochs,
            tuneNetwork = False,
            monitor_evaluation_cost = False,
            monitor_training_accuracy  =False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        n_data = len(self.evalDataList)
        n_test = len(self.testDataList)
        n = len(self.trainingDataList)

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        etaHistory = []
        j = 0
        noImprovement = 0
        currentEta = self.eta
        currentMiniBatchSize = self.miniBatchSize
#        for j in np.arange(epochs):
        while j < epochs:
            random.shuffle(self.trainingDataList)
            mini_batches = [self.trainingDataList[k:k + currentMiniBatchSize] for k in np.arange(0, n, currentMiniBatchSize)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, currentEta, self.lmbda, len(self.trainingDataList))

            if tuneNetwork:
                print("Epoch {} complete".format(j))
                if monitor_training_accuracy:
                    accuracy = self.accuracy(training = True)
                    training_accuracy.append(accuracy)
                    print("Accuracy on training data: {0} / {1}".format(accuracy, n))
                if monitor_evaluation_cost:
                    cost = self.total_cost(self.evalDataList, self.lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print("Cost on evaluation data: {}".format(cost))

                cost = self.total_cost(self.trainingDataList, self.lmbda)
                accuracy = self.accuracy(evaluation = True)
                last = len(training_cost)-1     # cost and accuracy increment in lock step so only looking at cost is OK

                if len(training_cost) == 0:     # on the first epoch we don't allow tuning
                    training_cost.append(cost)
                    evaluation_accuracy.append(accuracy)
                    etaHistory.append(currentEta)
                    j += 1
                    print("Cost on training data: {}".format(cost))
                    print("Accuracy on evaluation data: {0} / {1}".format(accuracy, n_data))
                else:

                    improvement = True
                    if self.priority == "trainingCost":
                        message = "Cost"
                        if cost > training_cost[last]:              # this cost higher than previous saved cost
                            improvement = False
                    else:
                        message = "Accuracy"
                        if accuracy < evaluation_accuracy[last]:    # this accuracy lower than previos save accuracy
                            improvement = False

                    if not improvement:
                        if noImprovement < 9:
                            noImprovement += 1
                        else:
                            return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, etaHistory

                        print("This cost {0}, previous saved cost {1}".format(cost, training_cost[last]))
                        print("This accuracy {0}, previous saved accuracy {1}".format(accuracy, evaluation_accuracy[last]))

                        self.weights = self.oldWeights.copy()
                        self.biases = self.oldBiases.copy()
                        # temp = (accuracy - evaluation_accuracy[lastAcuuracyIndex]) / evaluation_accuracy[lastAcuuracyIndex]
                        # eta = eta * (1 + 20 * temp)

                        print(message + " is worse")
                        if currentEta > self.eta/100:
                            # currentEta /= 2
                            currentMiniBatchSize = (currentMiniBatchSize * 8) // 10
                            print("Keeping old weights and biases and decreasing leaning rate to {}".format(currentEta))
                        else:
                            print("Keeping old weights, biases, and learning rate (it got too small)")


                        # if noImprovement < 10:
                        print("Stuck on this epoch {} times".format(noImprovement))
                        temp = len(self.trainingDataList)
                        self.trainingDataList = augmentData.augmentTrainingData2(self.trainingDataList,
                                                                                 self.evalDataList,
                                                                                 self.feedforward, initial = False)
                        print("Adding {} new training data trials".format(len(self.trainingDataList) - temp))
                        # else:
                        #     return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, etaHistory


                        # if noImprovement > 4:
                        #     if currentMiniBatchSize > 6:
                        #         currentMiniBatchSize = (currentMiniBatchSize * 8) // 10
                        #         print("Decreased mini-batch size to {}".format(currentMiniBatchSize))
                        #     else:
                        #         print("Can't decrease mini batch size, its down to {}".format((currentMiniBatchSize)))

                    else:   # training cost decreased
                        self.oldWeights = self.weights.copy()
                        self.oldBiases = self.biases.copy()
                        # temp = (accuracy - evaluation_accuracy[lastAcuuracyIndex]) / evaluation_accuracy[lastAcuuracyIndex]
                        # eta = eta * (1 + 5 * temp)
                        # currentEta *= 1.25
                        # currentMiniBatchSize = self.miniBatchSize    # go back to original minibatch size
                        currentMiniBatchSize = (currentMiniBatchSize * 10) // 8
                        print("This cost {0}, previous saved cost {1}".format(cost, training_cost[last]))
                        print("This accuracy {0}, previous saved accuracy {1}".format(accuracy, evaluation_accuracy[last]))
                        print(message + " is better")
                        print("Increasing the learning rate to {}".format(currentEta))
                        training_cost.append(cost)
                        evaluation_accuracy.append(accuracy)
                        etaHistory.append(currentEta)
                        noImprovement = 0
                        j += 1


                    # if evaluation_accuracy == len(self.evalDataList):
                    #     print("Success, exiting now!")
            else:
                j += 1

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, etaHistory

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in np.arange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, training = False, evaluation = False, test = False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        note: training data is a tuple of two vectors while the other data is a tuple of a vector and a scalar
        """
        results = list()

        if training:
            results = [(np.argmax(self.feedforward(x, useDropout = False)), np.argmax(y)) for (x, y) in self.trainingDataList]

        if evaluation:
            results = [(np.argmax(self.feedforward(x, useDropout = False)), y) for (x, y) in self.evalDataList]

        if test:
            results = [(np.argmax(self.feedforward(x, useDropout=False)), y) for (x, y) in self.testDataList]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def seeCosineSimilarityPatterns(self):

        stride = 7
        yes = ["Y", "y", "Yes", "yes"]
        similarityWeights = self.weights[0]

        # for numPattern in np.arange(4):
        #     for row in np.arange(4):
        #         for col in np.arange(4):
        #             selection = (numPattern *16) + (row * 4) + col
        #             temp = similarityWeights[selection, :]
        #             temp = temp.reshape((28,28))
        #             temp = temp[(row * stride) : ((row * stride) + stride), (col * stride) : ((col * stride) + stride)]
        #             if selection == 0:
        #                 plt.imshow(temp)
        #                 plt.colorbar()
        #                 plt.title("Pattern{0}, Row{1}, Col{2}".format(numPattern, row, col))
        #                 plt.pause(0.5)
        #                 myInput = input("See another?")
        #                 if myInput not in yes:
        #                     return
        #             else:
        #                 plt.close()
        #                 plt.imshow(temp)
        #                 plt.colorbar()
        #                 plt.title("Pattern{0}, Row{1}, Col{2}".format(numPattern, row, col))
        #                 plt.pause(0.5)
        #                 myInput = input("See another?")
        #                 if myInput not in yes:
        #                     return

        for numPattern in np.arange(4):
            myOutput = np.zeros((28, 28))
            for row in np.arange(4):
                for col in np.arange(4):
                    selection = (numPattern *16) + (row * 4) + col
                    temp = similarityWeights[selection, :]
                    temp = temp.reshape((28,28))
                    temp = temp[(row * stride): ((row * stride) + stride), (col * stride): ((col * stride) + stride)]

                    myOutput[(row * 7) : ((row * 7) + 7), (col * 7) : ((col * 7) + 7)] = temp

                    # temp = temp[(row * stride) : ((row * stride) + stride), (col * stride) : ((col * stride) + stride)]
                    # if selection == 0:
                    #     plt.imshow(temp)
                    #     plt.colorbar()
                    #     plt.title("Pattern{0}, Row{1}, Col{2}".format(numPattern, row, col))
                    #     plt.pause(0.5)
                    #     myInput = input("See another?")
                    #     if myInput not in yes:
                    #         return
                    # else:
                    #     plt.close()
                    #     plt.imshow(temp)
                    #     plt.colorbar()
                    #     plt.title("Pattern{0}, Row{1}, Col{2}".format(numPattern, row, col))
                    #     plt.pause(0.5)
                    #     myInput = input("See another?")
                    #     if myInput not in yes:
                    #         return
            plt.imshow(myOutput)
            plt.colorbar()
            plt.title("Pattern {}".format(numPattern))
            plt.pause(5.0)
            plt.close()
        return

    def augmentTrainingData(self, x, vectorY):

        xArray = x.reshape((28, 28))
#       plt.imshow(xArray)
#       plt.pause(5.0)
#       plt.close()

        # deform left side down 1, right side up 1
        temp = np.zeros((28,28))
        temp[2 : 15, 1 : 14] = xArray[1 : 14, 1 : 14]
        temp[15 : , 1 : 14] = xArray[14 : 27, 1 : 14]
        temp[13 : 26, 14 : 27] = xArray[14 : 27, 14 : 27]
        temp[ : 13, 14 : 27] = xArray[1 : 14, 14 : 27]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #left up 1, right down 1
        temp = np.zeros((28,28))
        temp[ : 13, 1: 14] = xArray[1: 14, 1: 14]
        temp[13 : 26, 1: 14] = xArray[14: 27, 1: 14]
        temp[15: , 14: 27] = xArray[14: 27, 14: 27]
        temp[2: 15, 14: 27] = xArray[1: 14, 14: 27]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #top right 1, bottom left 1
        temp = np.zeros((28,28))
        temp[1 : 14, 2: 15] = xArray[1: 14, 1: 14]
        temp[14 : 27, : 13] = xArray[14: 27, 1: 14]
        temp[14 : 27, 13: 26] = xArray[14: 27, 14: 27]
        temp[1 : 14, 15 : ] = xArray[1: 14, 14: 27]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #top left 1, bottom right 1
        temp = np.zeros((28,28))
        temp[1 : 14, : 13] = xArray[1: 14, 1: 14]
        temp[14 : 27, 2 : 15] = xArray[14: 27, 1: 14]
        temp[14 : 27, 15 : ] = xArray[14: 27, 14: 27]
        temp[1 : 14, 13 : 26] = xArray[1: 14, 14: 27]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #drop top quarter of rows
        temp = np.zeros((28,28))
        temp[7 : , : ] = xArray[7: , : ]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #drop lower right quadrant
        temp = np.zeros((28,28))
        temp[ : 14, : 14] = xArray[ : 14,  : 14]
        #missing quadrant
        temp[14 :  , 14 : ] = xArray[14 : , 14: ]
        temp[ : 14, 14 : ] = xArray[ : 14, 14 : ]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))

        #drop lower left quadrant
        temp = np.zeros((28,28))
        temp[ : 14,  : 14] = xArray[ : 14,  : 14]
        temp[14 : ,  : 14] = xArray[14 : ,  : 14]
        #missing quadrant
        temp[ : 14, 14 : ] = xArray[ : 14, 14 : ]
        # plt.imshow(temp)
        # plt.pause(5.0)
        # plt.close()
        newTest = temp.reshape((784,1))
        self.trainingDataList.append((newTest, vectorY))



        return

    def finalResults(self):
        failures = dict()
        results = list()
        histList = list()
        yes = ["y", "Y", "yes", "Yes"]

        for (x, y) in self.testDataList:
            currentResult = np.argmax(self.feedforward(x, useDropout=False))
            results.append((currentResult, y))

            if (currentResult != y):  # didn't match so save it
                histList.append(y)
                if y in failures.keys():
                    failures[y].append((currentResult, x))
                else:
                    failures[y] = list()
                    failures[y].append((currentResult, x))

        temp = sum(int(x == y) for (x, y) in results)
        print("The accuracy on the test data is {} / 10000".format(temp))

        plt.title("Histogram of Failed Test Data Trials")
        plt.hist(np.asarray(histList), 10)
        plt.pause(5.0)

        reply = input("Look at failed trials")

        while (reply in yes):
            reply = input("Which bad boy (0-9) do you want to see?")
            try:
                key = int(reply)
                print("There were {0} {1}s that failed".format(len(failures[key]), key))
                for num in np.arange(len(failures[key])):
                    thisFailure = failures[key][num][1]
                    valArray = thisFailure.reshape((28, 28))
                    plt.title("Is a {0}, catagorized as {1}".format(key, failures[key][num][0]))
                    plt.imshow(valArray)
                    plt.pause(5.0)
                    plt.close()
                    reply = input("See another example?")
                    if reply not in yes:
                        return
            except:
                print("You didnt enter a numeral")

            reply = ("Look at another number?")

        return
#******************************************************************************
#******************************************************************************
#******************************************************************************
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network2(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def buildSimilarityWeights():

    similarityWeights = np.ones((64,784))
    stride = 7

    vertical = np.array([[0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0]])


    horizontal = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])

    diagDown = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]])

    diagUp = np.array([[0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]])

    patterns = [vertical, horizontal, diagDown, diagUp ]

    for numPattern in np.arange(len(patterns)):
        for row in np.arange(4):
            for col in np.arange(4):
                base = np.zeros((28,28))
                base[(row * stride) : ((row * stride) + stride), (col * stride) : ((col * stride) + stride)] = patterns[numPattern]
                similarityWeights[((numPattern *16) + ((row * 4) + col)), :] = base.flatten()

    return similarityWeights

f = lambda value: 1 if value > 1 else value
fVect = np.vectorize(f)

#*********************************** Sandbox **********************************
# dropOut = 0.2
# test = np.ones((64,1))
# test *= np.random.binomial([np.ones(test.shape)], .80)[0]
# test += np.random.binomial([np.ones(test.shape)], .2)[0] * 0.2
# test = fVect(test)

# myfilters = convNet.initializeConvFilters(num=5)
# bias = convNet.initializeBias(myfilters)
#
# data = convNet.makeTestData()[0]
#
# myConvResult = convNet.conv(data, myfilters, bias, activation="relu")
# myPoolResult = convNet.maxPool(myConvResult, xStride=2, yStride=2)
#
# resultDims = myPoolResult.shape
#
# resultColVec = myPoolResult.reshape((resultDims[0] * resultDims[1] * resultDims[2], 1))

# myMatrix = np.array([[1, 2, 3],
#                     [1, 2, 3],
#                     [1, 2, 3]])
#
# myResult = (myMatrix * myMatrix).sum()


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network2([784, 64, 32, 10], cost = CrossEntropyCost, cosineSimilarity = True,
               miniBatchSize = 20, eta = 0.5, lmbda = 5.0, dropout = 0.1, priority = "evaluationAccuracy",
               training_data =  training_data, eval_data = validation_data, test_data = test_data)

net.initialAugmentation()

#net.seeCosineSimilarityPatterns()
#net.large_weight_initializer()

_, evaluation_accuracy, training_cost, _, etaHistory = net.SGD(40, tuneNetwork= True)
#net.seeCosineSimilarityPatterns()

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.set_title("Training Cost")
ax1.set_xlabel("Epoch Number")
ax1.set_ylabel("Cost")
ax1.plot(training_cost)
ax2.set_title("Evaluation Accuracy")
ax2.set_xlabel("Epoch Number")
ax2.set_ylabel("Accuracy")
ax2.plot(evaluation_accuracy)
ax3.set_title("Learning Rate")
ax3.set_xlabel("Epoch Number")
ax3.set_ylabel("Learning Rate (Eta)")
ax3.plot(etaHistory)
fig1.tight_layout(h_pad=0.5)
plt.show()


finalAccuracy = net.finalResults()

