
def getAbsMaxIndex(data_a, data_b):
    max_index = 0
    max_value = -1
    for i in range(len(data_a)):
        cur = abs(data_b[i] - data_a[i])
        if cur >= max_value:
            max_index = i

    return max_index
