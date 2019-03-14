import numpy as np
import matplotlib.pyplot as plt

def findBlockInfo (vec):
    n = 1
    blockSize = len(vec)
    numBlocks = 0
    stride = 0

    while (blockSize % 4) == 0:
        blockSize /= 4

    if blockSize == len(vec):
        print ("There's a problem with the length {} of the input vector".format(len(vec)))
        return (numBlocks, blockSize, stride)

    numBlocks = len(vec) / blockSize

    while (n * n) <= blockSize:
        if (blockSize % (n * n)) == 0:
            stride = n
        n += 1

    if n == 1:
        print("There's a problem with the length {} of the input vector".format(len(vec)))
        return (numBlocks, blockSize, stride)


    return (numBlocks, blockSize, stride)

def remapVec (vec, numBlocks, blockSize, stride):
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
#        plt.colorbar()
        plt.pause(5.0)

    return