import numpy as np

if __name__ == '__main__':
    a = [1, 3, 7, 9, 6]
    b = [7, 2, 5, 2, 9]

    c = np.array(a) - np.array(b)
    print(abs(c))