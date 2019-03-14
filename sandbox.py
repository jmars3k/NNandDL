import random as rnd
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dendriticCalc as dC

test = np.arange(784).reshape((784, 1))

numBlocks, blockSize, stride = dC.findBlockInfo(test)

myresult = dC.remapVec(test, numBlocks, blockSize, stride)

plt.plot(myresult)

dC.visualize(myresult, numBlocks)

