# Contineous CMAC
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Implementation of contineous CMAC to predict a sine wave.


import numpy as np
import matplotlib.pyplot as plt
import math


def ip_mapping(n, overlap, percent):
    i = math.floor(((n + 1) * mapping_index_end) / ip_vector)
    arr = np.zeros(ip_space)

    arr[i] = percent
    arr[(i+1): (i + overlap)] = np.ones(overlap - 1)
    if i < 31:
        arr[i+overlap] = 1 - percent

    return arr, i


def generate_data(res, test_no):
    ip_data = np.linspace(0, 2*3.14, res)
    op_data = np.sin(ip_data)
    ip_test = []
    op_test = []
    ip_train = []
    op_train = []
    train_indices = []
    test_indices = []

    idx = np.random.randint(0, res, int(test_no * res))
    for i in range(ip_vector):
        if i in idx:
            ip_test.append(ip_data[i])
            op_test.append(op_data[i])
            test_indices.append(i)
        else:
            ip_train.append(ip_data[i])
            op_train.append(op_data[i])
            train_indices.append(i)

    return [ip_data, op_data, ip_train, ip_test, op_train, op_test,
            train_indices, test_indices, res]


def accuracy(real, pred):
    result = np.count_nonzero(abs(pred - real) < error_th)
    acc = result / len(real)
    return acc


def train_CMAC(data, overlap):
    convergence = []
    for i in range(iterations):
        pre_w = weights
        for j in data[6]:
            percent = np.random.rand(1)
            map, idx = ip_mapping(j, overlap, percent)
            output = sum(np.multiply(map, weights))
            if idx < 31:
                error = (data[1][j] + (1-percent)*data[1][j+1]) - output
                correction = (learning_rate * error) / overlap

                weights[idx] = weights[idx] + percent * correction
                for k in range(overlap - 1):
                    weights[idx + k + 1] = weights[idx + k + 1] + correction
                weights[idx+overlap] = weights[idx + overlap] + \
                    (1 - percent) * correction

            else:
                error = data[1][j] - output
                correction = (learning_rate * error) / overlap
                for k in range(overlap):
                    weights[idx + k] = weights[idx + k] + correction
        convergence.append(np.dot(weights, pre_w.T)/35)
    return weights, convergence


def test_CMAC(weights):
    results = []

    for i in data[7]:
        map, idx = ip_mapping(i, overlap, 1)
        output = sum(np.multiply(map, weights))
        results.append(output)
    return results


if __name__ == "__main__":
    learning_rate = 0.2
    ip_space = 35
    overlap = 4
    ip_vector = 100
    iterations = 1000
    error_th = 0.05
    split = 0.3

    mapping_index_end = ip_space - overlap
    weights = np.random.rand(ip_space)

    data = generate_data(ip_vector, split)
    weights, convergence = train_CMAC(data, overlap)
    outputs = test_CMAC(weights)

    accuracy = accuracy(np.array(data[5]), np.array(outputs))
    print("accuracy is: ", accuracy)

    plt.figure()
    plt.plot(data[2], data[4], 'ro', label='Trained Data')
    plt.plot(data[3], outputs, 'bo', label='Real Output')
    plt.plot(data[3], data[5], 'go', label='Expected Output')
    plt.legend()

    plt.figure()
    plt.plot(range(1000), convergence, 'k-', label='Convergence')
    plt.legend()
    plt.show()
