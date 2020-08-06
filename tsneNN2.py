import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 输入为(n*m)的矩阵，表示n个样本，m个属性
# 返回一个距离矩阵
# numpy
def cal_pairwise_dist(x):
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    sum_x = np.sum(np.square(x), 1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    # 返回任意两个点之间距离的平方
    return dist


# tensorflow 2.0
def cal_pairwise_dist_tf(x):
    sum_x = tf.reduce_sum(tf.square(x), 1)
    dist = tf.math.add(tf.transpose(tf.math.add(-2 * tf.matmul(x, tf.transpose(x)), sum_x)), sum_x)
    # shape = (n, n)
    return dist


# 计算困惑度，最终会选择合适的beta，也就是每个点的方差啦
def cal_perplexity(dist, idx=0, beta=1.0):
    # '''计算perplexity, D是距离向量，
    # idx指dist中自己与自己距离的位置，beta是高斯分布参数
    # 这里的perp仅计算了熵，方便计算
    # '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        prob /= sum_prob
        perp = 0
        for pj in prob:
            if pj != 0:
                perp += -pj * np.log(pj)
    # 困惑度和pi\j的概率分布
    return perp, prob


def seach_beta(x, tol=1e-5, perplexity=30.0):
    # '''二分搜索寻找beta,并计算pairwise的prob
    # '''
    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf
        betamax = np.inf
        # dist[i]需要换不能是所有点
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))  # beta = 1 / sigma^2
    return beta


def cal_prob(x, beta):
    # x.shape = (n, d)
    # beta = (n, 1)
    (n, d) = x.shape
    # dist.shape = (n, n)
    dist = cal_pairwise_dist_tf(x)
    beta = tf.tile(beta, [1, n])
    dist = tf.cast(dist, dtype=tf.float32)
    beta = tf.cast(beta, dtype=tf.float32)
    pair_prob = tf.math.exp(- tf.multiply(dist, beta))
    mask = tf.zeros(n)
    pair_prob = tf.linalg.set_diag(pair_prob, mask)

    return pair_prob


def tsne_loss(x, y, beta):

    (n, d) = x.shape
    print(x.shape)

    P = cal_prob(x, beta)
    P = P + np.transpose(P)
    P = P / np.sum(P)  # pij
    P = np.maximum(P, 1e-12)

    # Compute pairwise affinities
    sum_y = tf.reduce_sum(tf.square(y), axis=1)
    num = 1 / (1 + tf.math.add(tf.transpose(tf.math.add(-2 * tf.matmul(y, tf.transpose(y)), sum_y)), sum_y))
    mask = tf.zeros(n)
    num = tf.linalg.set_diag(num, mask)
    Q = num / tf.reduce_sum(num)  # qij
    Q = tf.maximum(Q, 1e-12)  # X与Y逐位比较取其大者

    C = tf.reduce_sum(P * tf.math.log(P / Q))
    return C