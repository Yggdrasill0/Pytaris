import numpy as np
import time
import sys
from collections import OrderedDict, defaultdict
sys.setrecursionlimit(int(1E8))
from multiprocessing import Pool, cpu_count
import os
import argparse


#Construct lookup table, imported directly from Paul Bourke's site
# (Source: http://paulbourke.net/geometry/polygonise/)

tri_table =[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
            [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
            [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
            [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
            [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
            [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
            [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
            [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
            [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
            [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
            [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
            [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
            [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
            [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
            [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
            [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
            [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
            [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
            [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
            [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
            [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
            [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
            [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
            [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
            [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
            [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
            [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
            [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
            [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
            [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
            [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
            [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
            [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
            [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
            [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
            [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
            [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
            [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
            [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
            [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
            [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
            [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
            [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
            [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
            [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
            [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
            [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
            [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
            [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
            [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
            [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
            [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
            [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
            [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
            [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
            [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
            [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
            [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
            [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
            [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
            [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
            [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
            [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
            [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
            [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
            [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
            [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
            [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
            [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
            [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
            [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
            [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
            [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
            [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
            [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
            [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
            [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
            [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
            [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
            [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
            [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
            [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
            [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
            [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
            [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
            [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
            [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
            [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
            [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
            [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
            [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
            [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
            [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
            [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
            [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
            [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
            [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


periodic_table_translator = {1:"H", 6: "C", 12: "Mg", 13: "Al", 8: "O", 7: "N", 29: "Cu", 27: "Co", 17: "Cl", 
                             35 : "Br", 15 : "P", 16 :"S", 9:"F"}
cache={}

def interpolation_alpha_value(v1, v2, t):
    """
    Using a linear interpolation, computes the corresponding convex coefficient alpha in [0,1]
    accounting for the distance between two nodes in which the desired potential lies, taken from
    "https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb"
    """

    if v1 == v2 and t == v1:
        return 0.0
    if (t > v1 and t > v2) or (t < v1 and t < v2):
        return None
    return (v1 - t) / (v1 - v2)


def linear_interpolation(edge, cells, top, left, depth, thres):
  """Calculates the coordintates x,y,z based on the \alpha value, takes into account 
  the possible 12 Edge cases, taken from
  "https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb"
  """

  tval = 0
  point = None

  #Edge 0 case
  if (edge == 0):
    if (((left,top,depth),(left+1,top,depth)) in cache):
      point = cache[((left,top,depth),(left+1,top,depth))]


    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left+1,top,depth],thres)
      if (tval is None):
        return None
      point = (left+tval,top,depth)
      cache[((left,top,depth),(left+1,top,depth))] = point

    return point

  #Edge 1 case
  if (edge == 1):
    if (((left+1,top,depth),(left+1,top+1,depth)) in cache):
      point = cache[((left+1,top,depth),(left+1,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth],cells[left+1,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left+1,top+tval,depth)
      cache[((left+1,top,depth),(left+1,top+1,depth))] = point

    return point

  #Edge 2 case
  if (edge == 2):
    if (((left,top+1,depth),(left+1,top+1,depth)) in cache):
      point = cache[((left,top+1,depth),(left+1,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth],cells[left+1,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left+tval,top+1,depth)
      cache[((left,top+1,depth),(left+1,top+1,depth))] = point

    return point

  #Edge 3 case
  if (edge == 3):
    if (((left,top,depth),(left,top+1,depth)) in cache):
      point = cache[((left,top,depth),(left,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left,top+tval,depth)
      cache[((left,top,depth),(left,top+1,depth))] = point
    return point

  #Edge 4 case
  if (edge == 4):
    if (((left,top,depth+1),(left+1,top,depth+1)) in cache):
      point = cache[((left,top,depth+1),(left+1,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth+1],cells[left+1,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left+tval,top,depth+1)
      cache[((left,top,depth+1),(left+1,top,depth+1))] = point

    return point

  #Edge 5 case
  if (edge == 5):
    if (((left+1,top,depth+1),(left+1,top+1,depth+1)) in cache):
      point = cache[((left+1,top,depth+1),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth+1],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top+tval,depth+1)
      cache[((left+1,top,depth+1),(left+1,top+1,depth+1))] = point

    return point

  #Edge 6 case
  if (edge == 6):
    if (((left,top+1,depth+1),(left+1,top+1,depth+1)) in cache):
      point = cache[((left,top+1,depth+1),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth+1],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+tval,top+1,depth+1)
      cache[((left,top+1,depth+1),(left+1,top+1,depth+1))] = point

    return point

  #Edge 7 case
  if (edge == 7):
    if (((left,top,depth+1),(left,top+1,depth+1)) in cache):
      point = cache[((left,top,depth+1),(left,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth+1],cells[left,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top+tval,depth+1)
      cache[((left,top,depth+1),(left,top+1,depth+1))] = point

    return point

  #Edge 8 case
  if (edge == 8):
    if (((left,top,depth),(left,top,depth+1)) in cache):
      point = cache[((left,top,depth),(left,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top,depth+tval)
      cache[((left,top,depth),(left,top,depth+1))] = point

    return point

  #Edge 9 case
  if (edge == 9):
    if (((left+1,top,depth),(left+1,top,depth+1)) in cache):
      point = cache[((left+1,top,depth),(left+1,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth],cells[left+1,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top,depth+tval)
      cache[((left+1,top,depth),(left+1,top,depth+1))] = point

    return point

  #Edge 10 case
  if (edge == 10):
    if (((left+1,top+1,depth),(left+1,top+1,depth+1)) in cache):
      point = cache[((left+1,top+1,depth),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top+1,depth],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top+1,depth+tval)
      cache[((left+1,top+1,depth),(left+1,top+1,depth+1))] = point

    return point

  #Edge 11 case
  if (edge == 11):
    if (((left,top+1,depth),(left,top+1,depth+1)) in cache):
      point = cache[((left,top+1,depth),(left,top+1,depth+1))]
    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth],cells[left,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top+1,depth+tval)
      cache[((left,top+1,depth),(left,top+1,depth+1))] = point

    return point


def getContourCase(top,left,depth, thres,cells):
  """8-bit look up table to account for possible cases"""

  x = 0
  if (thres < cells[left,top+1,depth+1]):
    x = 128
  if (thres < cells[left+1,top+1,depth+1]):
    x = x + 64
  if (thres < cells[left+1,top,depth+1]):
    x = x + 32
  if (thres < cells[left,top,depth+1]):
    x = x + 16
  if (thres < cells[left,top+1,depth]):
    x = x + 8
  if (thres < cells[left+1,top+1,depth]):
    x = x + 4
  if (thres < cells[left+1,top,depth]):
    x = x + 2
  if (thres < cells[left,top,depth]):
    x = x + 1
  case_value = tri_table[x]

  return case_value


def process_threshold(args):
    """Vertex and faces array calculation using the geometric values provided in the gaussian
    cube (i.e origin of cube, x-direction step, y-direction step, z-direction step), modified
    from
    "https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb"
    """

    (threshold, cells, dx, dy, dz, origin) = args

    x0, y0, z0 = origin

    rows, cols, zcols = cells.shape
    vertex_counter = 0
    vertex_array = OrderedDict()
    face_array = []

    for left in range(rows - 1):
        for top in range(cols - 1):
            for depth in range(zcols - 1):
                case_val = getContourCase(top, left, depth, threshold, cells)
                if not case_val:
                    continue
                k = 0
                while case_val[k] != -1:
                    v1 = linear_interpolation(case_val[k], cells, top, left, depth, threshold)
                    v2 = linear_interpolation(case_val[k+1], cells, top, left, depth, threshold)
                    v3 = linear_interpolation(case_val[k+2], cells, top, left, depth, threshold)
                    k += 3

                    if v1 is None or v2 is None or v3 is None:
                        continue

                    face = [3, 0, 0, 0]
                    for i, v in enumerate([v1, v2, v3]):
                        if v not in vertex_array:
                            ix, iy, iz = v
                            real_x = x0 + ix*dx
                            real_y = y0 + iy*dy
                            real_z = z0 + iz*dz

                            vertex_array[v] = [vertex_counter, real_x, real_y, real_z]
                            vertex_counter += 1
                        face[i + 1] = vertex_array[v][0]
                    face_array.append(face)

    verts_np = np.array(list(vertex_array.values()))[:, 1:]
    faces_np = np.array(face_array)
    return threshold, verts_np, faces_np

def getContourSegments_parallel(cells, thresholds,
                                incremento_x, incremento_y, incremento_z,
                                origin):
    """Parallel calculation of faces and vertex array for each potential value,
    modified from
    "https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb"
    """
    
    t1 = time.time()
    print(min(cpu_count(), len(thresholds)))
    with Pool(processes=min(cpu_count(), len(thresholds))) as pool:
        args_list = [
            (t, cells, incremento_x, incremento_y, incremento_z, origin)
            for t in thresholds
        ]
        results = pool.map(process_threshold, args_list)

    output = {thres: {'verts': verts, 'faces': faces} for thres, verts, faces in results}

    t2 = time.time()
    print(f"\n[Parallel] Tiempo total de procesamiento: {t2 - t1:.3f} segundos ")
    return output


def obtain_coordinates(dir, neg):
  """Reads a gaussian-cube file and extracts the origin of the cube, the number of atoms, the 
   calculated MESP values """

  with open(dir, "r") as potential_cube:
    potential_cube_lines = potential_cube.readlines()

  coor_init_vals = list(map(float, potential_cube_lines[2].split()))
  atom_num, x0, y0, z0 = map(float, coor_init_vals[:1] + coor_init_vals[1:4])

  x_coor = list(map(float, potential_cube_lines[3].split()))
  points_x, delta_x = map(float, x_coor[:1] + x_coor[1:2])

  y_coor = list(map(float, potential_cube_lines[4].split()))
  points_y, delta_y = map(float, y_coor[:1] + y_coor[2:3])

  z_coor = list(map(float, potential_cube_lines[5].split()))
  points_z, delta_z = map(float, z_coor[:1] + z_coor[3:4])

  mesp_values = potential_cube_lines[6+int(atom_num):]
  mesp_values_f = np.array([float(value) for conj_valores in mesp_values for value in conj_valores.split()])

  min_value = np.min(mesp_values_f)
  mesp_values_f = mesp_values_f.reshape((int(points_x), int(points_y), int(points_z)))

  if neg:
    return mesp_values_f, delta_x, delta_y, delta_z, (x0, y0, z0), min_value
  else:
    return mesp_values_f, delta_x, delta_y, delta_z, (x0, y0, z0)


def generate_graph_from_faces(faces):
    """Creates the adjacency matrix of the nodes of the graph from the faces array"""

    graph = defaultdict(set)
    for tri in faces[:, 1:]:
        a, b, c = tri
        graph[a].update([b, c])
        graph[b].update([a, c])
        graph[c].update([a, b])

    return graph

def dfs(start, graph, visited):
    stack = [start]
    component = []
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        component.append(v)
        stack.extend(graph[v] - visited)

    return component


def extract_coordinates_cube_ordered_atoms(cube_file: os.path):
    with open(cube_file) as cube:
        lines = cube.readlines()

    atom_num = abs(int(lines[2].split()[0]))  
    start = 6                            
    end = start + atom_num

    coords = []
    atoms = []
    for line in lines[start:end]:
        atom,_ , x, y, z = map(float, line.split())
        coords.append([x, y, z])
        atoms.append(periodic_table_translator[atom])

    numbering_dic = {}
    atoms_with_numbers = []
    for atom in atoms:
        if atom not in numbering_dic.keys():
            numbering_dic[atom] = 0
        else:
            numbering_dic[atom] += 1
        atoms_with_numbers.append(f"{atom}{numbering_dic[atom] + 1}")

    return atoms_with_numbers, np.array(coords)


def nearest_atom(barycenter, atom_coords, atom_labels):

    diff = atom_coords - barycenter
    dists = np.linalg.norm(diff, axis=1)
    idx = np.argmin(dists)
    return atom_labels[idx]


def find_connected_components(graph):
    visited = set()
    components = []

    for v in graph:
        if v not in visited:
            components.append(dfs(v, graph, visited))

    return components


def faces_in_component(faces, component):
    """Returns the index of the faces completely cotained in the given component """

    comp = np.asarray(component)
    mask = np.all(np.isin(faces[:, 1:], comp), axis=1)
    return faces[mask][:, 1:]


def surface_properties(vert, face_idx):
    """Calculates the area, volume, centroid and the corresponding bounding box"""

    v0 = vert[face_idx[:, 0]]
    v1 = vert[face_idx[:, 1]]
    v2 = vert[face_idx[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    area = tri_area.sum()

    if area == 0:
        return 0.0, np.zeros(3), 0.0, None

   
    centroids = (v0 + v1 + v2) / 3.0
    barycenter = (centroids * tri_area[:, None]).sum(axis=0) / area

   
    volume = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0

    coords = vert[np.unique(face_idx)]
    bbox = np.array([
        coords[:, 0].min(), coords[:, 0].max(),
        coords[:, 1].min(), coords[:, 1].max(),
        coords[:, 2].min(), coords[:, 2].max()
    ])

    return area, barycenter, abs(volume), bbox


def inclusive_arange(start, stop, step):
    n = int(np.round((stop - start) / step))
    return start + step * np.arange(n + 1)


def obtain_values_neg(P_max, step, P_min=None, min_value=None):
    if P_min is None:
       P_min = np.ceil(min_value/step) * step
    return np.round(inclusive_arange(P_min, P_max, step), 3)


def obtain_values_pos(P_max, P_min, step):
    return np.round(inclusive_arange(P_max, P_min, -abs(step)),3)


def Pytaris(iso_values, mesp_values_f,
            delta_x, delta_y, delta_z,
            origin, atom_coords, atom_labels, tol=1e-5, negative=False):

    dicc_pytaris = {}
    conec = []

    if iso_values.size == 0:
        return np.empty((0, 3)), {}

    conec.append([iso_values[0], -1, -1])

    multi_data = getContourSegments_parallel(
        mesp_values_f,
        iso_values,
        delta_x, delta_y, delta_z,
        origin)

    bboxes_old = []
    representatives_old = []

    for value in iso_values:
        print(f"The current potential is {value}")

        if value not in multi_data:
            continue

        verts = multi_data[value]["verts"]
        faces = multi_data[value]["faces"]

        if verts.size == 0 or faces.size == 0:
            continue

        graph = generate_graph_from_faces(faces)
        components = find_connected_components(graph)

        component_rows = []
        component_atomtypes = []
        bboxes_new = []
        representatives_new = []

        for comp in components:
            face_idx = faces_in_component(faces, comp)
            area, barycenter, volume, bbox = surface_properties(verts, face_idx)

            if area < tol:
                continue

            atomtype = nearest_atom(barycenter, atom_coords, atom_labels)

            representative = (
                verts[face_idx[0, 0]] if negative else barycenter
            )

            component_rows.append(
                np.concatenate([[value, area, volume], barycenter], dtype= np.float64)
            )

            component_atomtypes.append(atomtype)
            bboxes_new.append(bbox)
            representatives_new.append(representative)

        if component_rows:
           dicc_pytaris[value] = {"data": np.vstack(component_rows), "atomtype": component_atomtypes}


        if bboxes_old:
            for i_new, bbox_new in enumerate(bboxes_new):
                xmin, xmax, ymin, ymax, zmin, zmax = bbox_new

                for i_old, rep_old in enumerate(representatives_old):
                    x, y, z = rep_old

                    if (xmin <= x <= xmax and
                        ymin <= y <= ymax and
                        zmin <= z <= zmax):
                        conec.append([value, i_old, i_new])

        bboxes_old = bboxes_new
        representatives_old = representatives_new

    return np.array(conec), dicc_pytaris


def preprocess_nodes(dicc_pytaris, reverse=False):
    keys = sorted(dicc_pytaris.keys(), reverse=reverse)
    keys = np.append(keys, 0)

    info_list = []
    atomtypes = []

    for k in keys[:-1]:  
        info_list.append(dicc_pytaris[k]["data"])
        atomtypes.extend(dicc_pytaris[k]["atomtype"]) 
    
    if not info_list or all(len(arr) == 0 for arr in info_list):
        info = np.zeros((1, 6))
        atomtypes = ["ROOT"]
        return keys, info, atomtypes

    info = np.vstack(info_list).reshape(-1, 6)

    
    info = np.vstack((info, np.zeros(6)))
    atomtypes.append("ROOT")

    return keys, info, atomtypes

def left_tag_nodes(nodes_data: np.array, keys: list) -> np.array:
  """
  Generates an integer tag that indicates the 3D position rank of a node for a "canonical" disposition of a graph,
  the priority of the function is given in anti-lexicographic order on the cartesian coordinates (i.e z > y >x)

  
  :param nodes_data: numpy array containing the information of each node in the order [potential, area, volume, x, y, z]
  :param keys: List containing the potential values swept during tree construction
  :returns np.ndarray, shape (M, 7) containing the selected nodes with an additional last column giving the canonical
      rank of each node within its potential group.
  """

  nodes_with_tag = []

  for key in keys:
    nodes_sel = nodes_data[np.isclose(nodes_data[:,0], key)] 

    if len(nodes_sel) != 0:
      
      tags = np.lexsort((nodes_sel[:, 3], nodes_sel[:, 4], nodes_sel[:, 5]))
      nodes_sel = np.hstack([nodes_sel, tags.reshape(-1,1)])
    
    nodes_with_tag.append(nodes_sel)
  
  return np.vstack(nodes_with_tag).reshape(-1,7)



def _nodes_per_potential(dicc_pytaris: dict):
    offsets = []
    total = 0
    for v in dicc_pytaris.values():
        
        offsets.append(total)
        total += len(v["data"])

    return np.array(offsets), total
  

def obtain_global_pre_edges(dicc_pytaris: dict, conec: np.array, ) -> list:
  """
  Returns a list consisting of tuples indicating the global indices of (origin_node, destination_node)
  
  :param dicc_pytaris: Dictionary of dictioniaries returned by the Pytaris function
  :param conec: np.array containing the edge information locally (current_isopot value, old_isopot_idx, current_isopot_idx)
  """

  edge_list = []
  conec = conec.reshape(-1, 3)[1:]
  keys = np.array(list(dicc_pytaris.keys()))
  offsets, num_nodes = _nodes_per_potential(dicc_pytaris)

  for isopotential, idx_old, idx_new in conec:
     try:
        pot_idx = np.where(np.isclose(keys, isopotential))[0][0] #anki, how to use np.where
        or_node_num   = offsets[pot_idx - 1] + int(idx_old)
        dest_node_num = offsets[pot_idx]     + int(idx_new)

        edge_list.append((int(or_node_num), int(dest_node_num)))
        
     except Exception as e:
        raise ValueError(f"There was an error in the generation of an edge {e}") 
  
  #All nodes at the last potential are connected by an artifitial node

  if len(keys) > 1:
    second_last_idx = len(keys) - 1
    root_node = num_nodes 

    for i in range(len(dicc_pytaris[keys[second_last_idx]]["data"])):
      or_node_num = offsets[second_last_idx] + i
      edge_list.append((int(or_node_num), int(root_node)))

  return edge_list


def obtain_global_edges(edge_list: list, nodes_data: np.array):
    """
    If a node has multiple outgoing edges, keep only the one whose destination
    is spatially closest (Euclidean distance in columns 3:6 of info_nodos).
    
    :param edge_list: List of global edges 
    :param nodes_data: numpy array containing the information of each node in the order [potential, area, volume, x, y, z]
    """

    if len(edge_list) == 0:
        return edge_list

    edges = np.array(edge_list, dtype=int)
    origins, counts = np.unique(edges[:, 0], return_counts=True)
    keep_mask = np.ones(len(edges), dtype=bool)

    for node in origins[counts > 1]:
      
        idxs = np.where(edges[:, 0] == node)[0]
        origin_xyz = nodes_data[node, 3:6]
        dest_xyz = nodes_data[edges[idxs, 1], 3:6]

        #distance_calculation l2
        diff = dest_xyz - origin_xyz
        dists = np.sqrt(np.sum(diff**2, axis=1))
        best_local = np.argmin(dists)

        to_remove = np.ones(len(idxs), dtype=bool)
        to_remove[best_local] = False

        keep_mask[idxs[to_remove]] = False

    edges = edges[keep_mask]
    return edges.tolist()


def create_file(file_name: str, nodes_data: np.array, 
                atom_types: np.array, edge_list: list):

  with open(file_name, "w") as fo:
    fo.write("graph [ \n")
    fo.write("directed 1 \n")
    for i in range(nodes_data.shape[0]):
      fo.write("node [ \n")
      fo.write(f"id {i} \n")
      fo.write(f'label "{i}" \n')
      fo.write(f"potentialValue {nodes_data[i][0]} \n")
      fo.write(f"areaValue {nodes_data[i][1]} \n")
      fo.write(f"volumeValue {nodes_data[i][2]} \n")
      fo.write(f'atomtype "{atom_types[i]}" \n')
      fo.write(f"xValue {nodes_data[i][3]} \n")
      fo.write(f"yValue {nodes_data[i][4]} \n")
      fo.write(f"zValue {nodes_data[i][5]} \n")
      fo.write(f"izquierda {int(nodes_data[i][6])} \n")
      fo.write("\t ] \n")

    for edge in edge_list:
      fo.write("edge [ \n")
      fo.write(f"source {edge[0]} \n")
      fo.write(f"target {edge[1]} \n")
      fo.write("\t ] \n")
    fo.write("tree 1 \n")
    fo.write("] \n")
