import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import Isomap
import tensorflow as tf
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import decomposition
import os

def loadmnist():
    '读取mnist数据集'''
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train.reshape((-1, 784)).astype("float32")
    x_test = x_test.reshape((-1, 784)).astype("float32")

    X = x_train[-5000:]
    y = y_train[-5000:]

    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (x_train, y_train)).shuffle(10000).batch(1000)

    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return X, y

def visualize(X,y):
    '嵌入空间可视化'''
    x_min, x_max = X.min(0), X.max(0)
    X_norm = (X - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./images/%s.jpg"%(sys._getframe().f_back.f_code.co_name))
    plt.show()

def tSNE():
    X, y = loadmnist()
    X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(X)
    visualize(X_tsne, y)

def isomap():
    X,y = loadmnist()
    X_isomap = Isomap(n_components=2).fit_transform(X)
    visualize(X_isomap, y)

def LLE():
    X,y = loadmnist()
    X_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(X)
    visualize(X_LLE, y)

def PCA():
    X,y = loadmnist()
    X_PCA = decomposition.PCA(n_components=2).fit_transform(X)
    visualize(X_PCA, y)

#
# def main(argv=None):
#     tSNE()
#     isomap()
#     LLE()
#     PCA()
#


def test_tsneNN(logits, y, path):
    for i in range(10, 40):
        perp = i
        X = manifold.TSNE(n_components=2, perplexity=perp, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(logits)

        x_min, x_max = X.min(0), X.max(0)
        X_norm = (X - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path + "\\{}.png".format(perp))




