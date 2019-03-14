import numpy as np
from numpy import linalg as LA

def initializeConvFilters(num = 4):

    convFilters = np.zeros((num, 5, 5))

    vertical = np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]])
    vertical = vertical / LA.norm(vertical)

    horizontal = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
    horizontal = horizontal / LA.norm(horizontal)

    diagDown = np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1]])
    diagDown = diagDown / LA.norm(diagDown)

    diagUp = np.array([[0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0]])
    diagUp = diagUp / LA.norm(diagUp)

    for filterNum in np.arange(num):
        if filterNum == 0:
            convFilters[0, :, :] = vertical
        elif filterNum == 1:
            convFilters[1, :, :] = horizontal
        elif filterNum == 2:
            convFilters[2, :, :] = diagDown
        elif filterNum == 3:
            convFilters[3, :, :] = diagUp
        else:
            temp = np.random.randn(5,5)
            temp = temp / LA.norm(temp)
            convFilters[filterNum, : , :] = temp

    return convFilters

def initializeBias(filters):

    filterDims = filters.shape
    numFilters = filterDims[0]
    bias = np.random.randn(numFilters, 1)
    return bias

def makeTestData():

    result = list()
    f = 1.0
    h = 0.5
    q = 0.25
               #      1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
    zeroA= np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, h, h, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, f, f, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, q, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, q, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, 0, 0, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, 0, h, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, h, f, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, f, h, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 17
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, 0, 0, 0, 0, 0, 0, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
                     [0, 0, 0, 0, 0, 0, 0, 0, h, f, f, q, 0, 0, 0, 0, q, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0],  # 19
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, q, 0, 0, q, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, f, f, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, f, f, f, f, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 22
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, h, h, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 23
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 24
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 25
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 26
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 27
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # 28


    result.append(zeroA)

    return result

def activationFun(type, data):

    relu = lambda x: x if x > 0 else 0

    if type == "sigmoid":
        return 1.0 / (1.0 + np.exp(-data))

    elif type == "relu":
        return relu(data)

    else:
        print("unkown activation function")
        return 0

def conv(data, filter, bias, activation = "sigmoid"):

    filterDims = filter.shape
    numFilters = filterDims[0]
    filterRows = filterDims[1]
    filterCols = filterDims[2]

    dataDims = data.shape
    dataRows = dataDims[0]
    dataCols = dataDims[1]

    j = (dataRows - filterRows) + 1
    k = (dataCols - filterCols) + 1

    result = np.zeros((numFilters, j, k))

    for filterNum in np.arange(numFilters):
        for row in np.arange(j):
            for col in np.arange(k):
                dataSubMatrix = data[row : (row + filterRows), col : (col + filterCols)]
                temp = (dataSubMatrix * filter[filterNum, : , :]).sum() + bias[filterNum]
                result[filterNum, row, col] = activationFun(type=activation, data=temp)

    return result

def maxPool(data, xStride, yStride):

    dataDims = data.shape
    numLayers = dataDims[0]
    layerRows = dataDims[1]
    layerCols = dataDims[2]

    if (layerRows % yStride) != 0:
        print("Vertical stride doesn't go into conv layer number of rows evenly")
        return

    if (layerCols % xStride) != 0:
        print("Horizontal stride doesn't go into conv layer number of columns evenly")
        return

    resultRows = layerRows // yStride
    resultCols = layerCols // xStride
    result = np.zeros((numLayers, resultRows, resultCols))

    for layer in np.arange(numLayers):
        for row in np.arange(resultRows):
            for col in np.arange((resultCols)):
                temp = data[layer, row * yStride : (row * yStride) + yStride, col * xStride : (col * xStride) + xStride]
                result[layer, row, col] = temp.max()

    return result