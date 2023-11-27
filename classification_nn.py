import random

import cv2
import os
import numpy as np



def read_images(parent_path):
    images, labels = [], []
    folders = {"buildings": 0, "forest": 1, "glacier": 2, "mountain": 3, "sea": 4, "street": 5}
    flag = 0
    for filename in os.listdir(parent_path):
        if filename.endswith('.jpg'):
            flag = 1
            pic = cv2.imread(os.path.join(parent_path, filename))
            pic = cv2.resize(pic, (30, 30))
            pic = cv2.normalize(pic, pic, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            gray = gray.reshape(30 * 30)
            images.append(gray)
        else:
            flag = 2
            path = parent_path + filename + "/"
            for img in os.listdir(path):
                pic = cv2.imread(os.path.join(path, img))
                pic = cv2.resize(pic, (30, 30))
                pic = cv2.normalize(pic, pic, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                images.append(gray.reshape(30 * 30))
                labels.append(folders[filename])
    images = np.array(images)
    labels = np.array(labels)
    if flag == 1:
        return images
    elif flag == 2:
        d = list(zip(images, labels))
        random.shuffle(d)
        images, labels = zip(*d)
        return images, labels


path_dev = 'seg_dev/seg_dev/'
dev_images, dev_labels = read_images(path_dev)
path_train = 'seg_train/seg_train/'
train_images, train_labels = read_images(path_train)
path_test = 'seg_test/'
test_images = read_images(path_test)


def activation(func, z):
    # z is the wi*xi+b
    if func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    elif func == "relu":
        return np.maximum(0, z)
    elif func == "softmax":
        # return np.exp(x) /  np.sum(np.exp(x))  #not stable
        return np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))


def derivative(func, x):
    if func == "sigmoid":
        a = (activation("sigmoid", x))
        return a * (1 - a)
    elif func == "relu":
        return 1 * (x > 0)
    elif func == "softmax":
        x = activation("softmax", x)
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    elif func == "ce":
        return -1 / x * np.log(2)
    elif func == "se":
        return -1 + x


def objective(func, x, y):
    # x is the output matrix from activation
    # y is the expected result
    if func == "ce":
        log = -1 * (np.log2(x[y]))
        return log
    elif func == "se":
        return 0.5 * (1 - x[y]) ** 2


def forward(features, w, b):
    sums = []
    active = []
    for i in range(6):
        sum = np.sum(np.dot(w[i].T, features))+b
        sums.append(sum)
        oi = activation("sigmoid", sums[i])
        active.append(oi)
    return sums, active

def backward(features, label,active, w, b, lr):
    #  loss += objective("ce", active, label)
    error = objective("se", active, label)
    dcost_dpred = error
    dpred_dc = derivative("sigmoid", active[label])
    z_delta = dcost_dpred * dpred_dc
    for i in range(6):
        w[i] -= lr * np.dot(features, z_delta)
    b -= lr * z_delta
    return w, b, error


def train(input, label, epoch, batch, lr):
    weights = np.random.rand(6, 900) * 0.01  # 6 weights for every pixel of ONE image
    bias = np.random.rand() * 0.01
    error_before = float("inf")
    for i in range(epoch):
        for j in range(len(input)):
            sum, active_sum = forward(input[j], weights, bias)
            if j != 0 and j % batch == 0:
                weights, bias, error = backward(input[j], label[j], active_sum, weights, bias, lr)
        if error < error_before:
                error_before = error
                np.savetxt('data.csv', weights, delimiter=',')
        print(error, error_before)


train(train_images, train_labels, 20, 16, 0.005)
