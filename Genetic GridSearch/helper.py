import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import mean_absolute_percentage_error as mape_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score




"""Helpful functions:
1. Timsort, insertion, merge
2. Merge dictionaries
"""


MIN_MERGE = 32


def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.

    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(pop, left, right, dir):
    for i in range(left + 1, right + 1):
        j = i

        if dir == 1:       ## Ascendimg
            while j > left and pop[j] < pop[j - 1]:
                pop[j], pop[j - 1] = pop[j - 1], pop[j]
                j -= 1
        else:               ## Descending
            while j > left and pop[j] > pop[j - 1]:
                pop[j], pop[j - 1] = pop[j - 1], pop[j]
                j -= 1


# Merge function merges the sorted runs
def merge(pop, l, m, r, dir = 1):

    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(pop[l + i])
    for i in range(0, len2):
        right.append(pop[m + 1 + i])

    i, j, k = 0, 0, l

    # after comparing, we merge those two array
    # in larger sub array
    if dir == 1:   ## Ascending
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                pop[k] = left[i]
                i += 1

            else:
                pop[k] = right[j]
                j += 1

            k += 1
    else:           ## Descending
        while i < len1 and j < len2:
            if left[i] >= right[j]:
                pop[k] = left[i]
                i += 1

            else:
                pop[k] = right[j]
                j += 1

            k += 1

    # Copy remaining elements of left, if any
    while i < len1:
        pop[k] = left[i]
        k += 1
        i += 1

    # Copy remaining element of right, if any
    while j < len2:
        pop[k] = right[j]
        k += 1
        j += 1

# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def tim_sort(pop, dir = 1):
    """
    Purpose:
    Sorts population based on fitness values

    Input: 
    pop: Takes in a population
    dir: '1' - Ascending, '0' - 'Descending'
    """
    n = len(pop)
    minRun = calcMinRun(n)

    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(pop, start, end, dir)

    # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:

        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):

            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(pop, left, mid, right)

        size = 2 * size

    return pop

def merge_dicts(dict1, dict2):
    # Check for overlapping keys
    intersection = set(dict1.keys()) & set(dict2.keys())
    if intersection:
        raise ValueError(f"Overlapping keys found: {intersection}")

    # Merge the dictionaries
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    
    return merged_dict


def evaluate(actual, predicted, metric):
    '''Returns desired metrics'''
    try:    ## Classification Problem
        metrics = {
                    "accuracy": accuracy_score(actual, predicted),
                    "f1": f1_score(actual, predicted, average = "weighted"),
                    "recall": recall_score(actual, predicted, average = "weighted"),
                    "precision": precision_score(actual, predicted, average = "weighted")
                    }

    except ValueError:  ## Regression Problem
        metrics = {
                    "r2":r2_score(actual, predicted), 
                    "rmse":mse_score(actual, predicted)**0.5,
                    "mse":mse_score(actual, predicted), 
                    "mae":mae_score(actual, predicted)
                    }


    return metrics[metric]

