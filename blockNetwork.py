import numpy as np
import matplotlib.pyplot as plt

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

class blockNetwork(object):
    """

    """

    def __init__(self, sizes, cost = CrossEntropyCost,
                 miniBatchSize=10, eta=0.5, lmbda=0.0, dropout=0, priority="trainingCost",
                 training_data=None, eval_data=None, test_data=None):

        self.num_layers = len(sizes)
        self.sizes = sizes

        self.numBlocks, self.blockSize, self.stride = self.findBockInfo()

        self.cost = cost
        self.miniBatchSize = miniBatchSize
        self.eta = eta
        self.lmbda = lmbda
        self.dropout = dropout
        self.priority = priority
        self.trainingDataList = list(training_data)
        self.evalDataList = list(eval_data)
        self.testDataList = list(test_data)

    def findBockInfo (self):
        """

        :return:
        """
        n = 1
        blockSize = self.sizes[0]   #initialize to 1 block same size as input layer
        numBlocks = 0
        stride = 0

        while (blockSize % 4) == 0:
            blockSize /= 4

        if blockSize == len(self.sizes[0]):
            print ("There's a problem with the length {} of the input vector".format(len(vec)))
            return (numBlocks, blockSize, stride)

        numBlocks = len(self.sizes[0]) / blockSize

        while (n * n) <= blockSize:
            if (blockSize % (n * n)) == 0:
                stride = n
            n += 1

        if n == 1:
            print("There's a problem with the length {} of the input vector".format(len(vec)))
            return (numBlocks, blockSize, stride)


        return (numBlocks, blockSize, stride)

    def remapVec (self, vec, numBlocks, blockSize, stride):
        """
        Remaps an input vector into an output vector with numBlocks of subblocks.  If the input vector was reshaped as a
        square matrix of numBlocks square blocks each subblock is reshaped as a vector and appended in order to generate a
        new output vector.
        :param vec: trial data in vector form
        :param numBlocks: number of subblocks you want to remap to
        :param blockSize: number of elements in each block
        :param stride: number of elements in each direction for subblock, subblock is assummed square
        :return:
        """

        majorStride = np.sqrt(len(vec)) #assumes the input is from a square matrix
        result = np.zeros(len(vec)).reshape((len(vec), 1))

        for idx1 in np.arange(numBlocks):
            for idx2 in np.arange(stride):

                temp = ((idx1 // 4) * stride + idx2) * majorStride
    #            print("Input Row starts with {0}".format(temp))

                low = int(temp + (idx1 % 4) * stride)
                high = low + stride
    #            print("Block {0}, Row {1}: IN low {2}, high{3}".format(idx1, idx2, low, high))
                temp = vec[low : high]

                low = int((idx1 * stride + idx2) * stride)
                high = low + stride
    #            print("Block {0}, Row {1}:  OUT low {2}, high {3}".format(idx1, idx2, low, high))
                result[low : high] = temp

        return result

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        layer 0 is input layer

        Layer 1 will have: self.numBlocks weight vectors all of size {1, self.blockSize)
                           bias vector (self.numBlocks, 1)

        Layer 2 will have: weight matrix (outputLayerSize, numBlocks)
                           bias vector (outputLayerSize, 1)

        """
        sizes = [self.sizes[0], self.numBlocks, self.sizes[1]]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.oldBiases = self.biases.copy()

        self.L1W = [np.random.randn(1, self.blockSize)
                        for idx in np.arange(self.numBlocks)]
        self.oldWeights = self.weights.copy()

def visualize(vec, numBlocks):
    """

    :param vec:
    :param numBlocks:
    :return:
    """

    blockSize = int(len(vec) / numBlocks)
    stride = int(np.sqrt(blockSize))
    start = 0

    for block in np.arange(numBlocks):
        temp = vec[start : start + blockSize]
        temp = temp.reshape((stride, stride))
        start += blockSize
        plt.imshow(temp)
        plt.pause(5.0)

    return