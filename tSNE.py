# tensorflow 1.14 version
import numpy as np
import tensorflow as tf


def cal_pairwise_dist(x):
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    # sum_x = np.sum(np.square(x), 1)
    sum_x = tf.math.reduce_sum(tf.math.square(x), axis=1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    # dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    dist = tf.math.add(tf.transpose(tf.math.add(-2 * tf.matmul(x, tf.transpose(x)), sum_x)), sum_x)
    #返回任意两个点之间距离的平方
    return dist


# 计算困惑度，最终会选择合适的beta，也就是每个点的方差啦
def cal_perplexity(dist, idx=0, beta=1.0):
    # '''计算perplexity, D是距离向量，
    # idx指dist中自己与自己距离的位置，beta是高斯分布参数
    # 这里的perp仅计算了熵，方便计算
    # '''
    # prob = np.exp(-dist*beta)
    prob = tf.math.exp(-dist * beta)
    # 设置自身prob为0
    # prob[idx] = 0
    mask = np.ones((100,), dtype=np.float32)
    mask[idx] = 0.0
    mask = tf.Variable(mask)
    prob = tf.multiply(prob, mask)

    # sum_prob = np.sum(prob)
    sum_prob = tf.reduce_sum(prob)
    if sum_prob == 0:
        # prob = np.maximum(prob, 1e-12)
        prob = tf.math.maximum(prob, 1e-12)
        perp = -12
    else:
        prob /= sum_prob
        perp = 0
        perp = tf.reduce_sum(tf.map_fn(lambda x: -x * tf.math.log(x) if x != 0 else 0, prob))
        # for pj in prob:
        #     if pj != 0:
        #         # perp += -pj*np.log(pj)
        #         perp += -pj*tf.math.log(pj)
    # 困惑度和pi\j的概率分布
    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):
    # '''二分搜索寻找beta,并计算pairwise的prob
    # '''
    # 初始化参数
    print("Computing pairwise distances...")
    # (n, d) = tf.shape(x)
    n = 100
    dist = cal_pairwise_dist(x)
    # pair_prob = np.zeros((n, n))
    pair_prob = tf.zeros((n, n), dtype=tf.float32)
    # beta = np.ones((n, 1))
    beta = tf.ones((n, 1), dtype=tf.float32)
    # 取log，方便后续计算
    # base_perp = np.log(perplexity)
    base_perp = tf.math.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = tf.constant(-np.inf)
        betamax = tf.constant(np.inf)
        #dist[i]需要换不能是所有点
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0




        def body1(beta, betamin,betamax):
            new_betamin = lambda: tf.Variable(beta[i], dtype=tf.float32)
            if betamax == np.inf or betamax == -np.inf:
                t0 = beta[:i]
                t1 = tf.expand_dims(beta[i][0] * 2)
                t1 = tf.expand_dims(t1, 0)
                t2 = beta[i+1:]
                new_beta = tf.concat([t0, t1, t2], 0)
                # beta[i].assign(beta[i] * 2)
            else:
                t0 = beta[:i]
                t1 = tf.expand_dims((beta[i][0] + betamax) / 2, 0)
                t1 = tf.expand_dims(t1, 0)
                t2 = beta[i + 1:]
                new_beta = tf.concat([t0, t1, t2], 0)
                # beta[i].assign((beta[i] + betamax) / 2)
            return new_beta

        def body2(beta, betamin,betamax):
            new_betamax = lambda: tf.Variable(beta[i], dtype=tf.float32)
            if betamin == np.inf or betamin == -np.inf:
                t0 = beta[:i]
                t1 = tf.expand_dims(beta[i][0] / 2, 0)
                t1 = tf.expand_dims(t1, 0)
                t2 = beta[i + 1:]
                new_beta = tf.concat([t0, t1, t2], 0)
                # beta[i].assign(beta[i] / 2)
            else:
                t0 = beta[:i]
                t1 = tf.expand_dims((beta[i][0] + betamin) / 2, 0)
                t1 = tf.expand_dims(t1, 0)
                t2 = beta[i + 1:]
                new_beta = tf.concat([t0, t1, t2], 0)
                # beta[i].assign((beta[i] + betamin) / 2)
            return new_beta

        def cond(perp_diff, tries, beta, betamin, betamax):
            return tf.cond(tf.math.logical_and(tf.math.abs(perp_diff) > tol, tries < 50), lambda: tf.constant(True), lambda: tf.constant(False))

        def body(perp_diff, tries, beta, betamin, betamax):
            beta = tf.cond(perp_diff > 0, lambda: body1(beta, betamin, betamax), lambda: body2(beta, betamin, betamax))
            betamin = tf.cond(perp_diff > 0, lambda: beta[i], lambda: betamin)
            betamax = tf.cond(perp_diff > 0, lambda: betamax, lambda: beta[i])
            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
            return perp_diff, tries, beta, betamin, betamax

        final_var = tf.while_loop(cond, body, [perp_diff, tries, beta, betamin, betamax],
                                 shape_invariants=[perp_diff.get_shape(), float,
                                                  tf.TensorShape([100, 1]),
                                                   tf.TensorShape([1, ]),
                                                   tf.TensorShape([1, ])])
        perp_diff = final_var[0]
        tries = final_var[1]
        beta = final_var[2]
        betamin = final_var[3]
        betamax = final_var[4]





        # while tf.math.abs(perp_diff) > tol and tries < 50:
        #     if
        #     if perp_diff > 0:
        #         betamin = beta[i].copy()
        #         if betamax == np.inf or betamax == -np.inf:
        #             beta[i] = beta[i] * 2
        #         else:
        #             beta[i] = (beta[i] + betamax) / 2
        #     else:
        #         betamax = beta[i].copy()
        #         if betamin == np.inf or betamin == -np.inf:
        #             beta[i] = beta[i] / 2
        #         else:
        #             beta[i] = (beta[i] + betamin) / 2
        #
        #     # 更新perb,prob值
        #     perp, this_prob = cal_perplexity(dist[i], i, beta[i])
        #     perp_diff = perp - base_perp
        #     tries = tries + 1







        # 记录prob值
        pair_prob[i,] = this_prob
    # print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    print("Mean value of sigma: ", tf.reduce_mean(tf.math.sqrt(1 / beta)))
    #每个点对其他点的条件概率分布pi\j
    return pair_prob


def tsne_loss(x, y,  no_dims=2, perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    (n, d) = x.shape
    print(x.shape)

    P = seach_prob(x, 1e-5, perplexity)
    # P = P + np.transpose(P)
    P = P + tf.transpose(P)
    # P = P / np.sum(P)  # pij
    P = P / tf.reduce_sum(P)  # pij
    # early exaggeration
    # pi\j
    # P = np.maximum(P, 1e-12)
    P = tf.math.maximum(P, 1e-12)

    # sum_y = np.sum(np.square(y), 1)
    sum_y = tf.reduce_sum(tf.math.square(y), axis=1)
    # num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
    num = 1 / (1 + tf.math.add(tf.math.add(tf.transpose(-2 * tf.matmul(y, tf.transpose(y)), sum_y)), sum_y))
    num[range(n), range(n)] = 0
    # Q = num / np.sum(num)  # qij
    Q = num / tf.reduce_sum(num)  # qij
    # Q = np.maximum(Q, 1e-12)  # X与Y逐位比较取其大者
    Q = tf.math.maximum(Q, 1e-12)  # X与Y逐位比较取其大者

    # Compute current value of cost function
    # C = np.sum(P * np.log(P / Q))
    C = tf.reduce_sum(P * tf.math.lof(P / Q))
    print("Iteration ", (iter + 1), ": loss is ", C)

    return C



# version 2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
#
# # 输入为(n*m)的矩阵，表示n个样本，m个属性
# # 返回一个距离矩阵
# def cal_pairwise_dist(x):
#     sum_x = tf.math.reduce_sum(tf.math.square(x), axis=1)
#     dist = tf.math.add(tf.transpose(tf.math.add(-2 * tf.matmul(x, tf.transpose(x)), sum_x)), sum_x)
#     return dist
#
#
# # 计算困惑度，最终会选择合适的beta，也就是每个点的方差啦
# def cal_perplexity(dist, idx=0, beta=1.0):
#     prob = tf.math.exp(-dist * beta)
#     n = prob.shape
#     # 设置自身prob为0
#     # prob[idx] = 0
#     mask = np.ones((n), dtype=np.float32)
#     mask[idx] = 0.0
#     mask = tf.Variable(mask)
#     prob = tf.multiply(prob, mask)
#
#     # sum_prob = np.sum(prob)
#     sum_prob = tf.reduce_sum(prob)
#     if sum_prob == 0:
#         # prob = np.maximum(prob, 1e-12)
#         prob = tf.math.maximum(prob, 1e-12)
#         perp = -12
#     else:
#         prob /= sum_prob
#         perp = 0
#         perp = tf.reduce_sum(tf.map_fn(lambda x: -x * tf.math.log(x) if x != 0 else 0, prob))
#     return perp, prob
#
#
# def seach_prob(x, tol=1e-5, perplexity=30.0):
#     # '''二分搜索寻找beta,并计算pairwise的prob
#     # '''
#     # 初始化参数
#     print("Computing pairwise distances...")
#     (n, d) = x.shape
#     dist = cal_pairwise_dist(x)
#     pair_prob = np.zeros((n, n))
#     beta = tf.ones((n, 1))
#     # 取log，方便后续计算
#     base_perp = tf.math.log(perplexity)
#
#     for i in range(n):
#         if i % 500 == 0:
#             print("Computing pair_prob for point %s of %s ..." % (i, n))
#
#         betamin = -np.inf
#         betamax = np.inf
#         # dist[i]需要换不能是所有点
#         perp, this_prob = cal_perplexity(dist[i], i, beta[i])
#
#         # 二分搜索,寻找最佳sigma下的prob
#         perp_diff = perp - base_perp
#         tries = 0
#         while tf.math.abs(perp_diff) > tol and tries < 50:
#             if perp_diff > 0:
#                 betamin = beta[i]
#                 if betamax == np.inf or betamax == -np.inf:
#                     mask1 = np.ones((n, 1), dtype=np.float32)
#                     mask1[i] = 0
#                     mask1 = tf.Variable(mask1)
#                     mask2 = np.zeros((n, 1), dtype=np.float32)
#                     mask2[i] = beta[i] * 2
#                     mask2 = tf.Variable(mask2)
#                     beta = tf.add(tf.multiply(beta, mask1), mask2)
#                     # beta[i] = beta[i] * 2
#                 else:
#                     mask = np.ones((n, 1), dtype=np.float32)
#                     mask[i] = 0
#                     mask = tf.Variable(mask)
#                     mask2 = np.zeros((n, 1), dtype=np.float32)
#                     mask2[i] = (beta[i] + betamax) / 2
#                     mask2 = tf.Variable(mask2)
#                     beta = tf.add(tf.multiply(beta, mask), mask2)
#                     # beta[i] = (beta[i] + betamax) / 2
#             else:
#                 betamax = beta[i]
#                 if betamin == np.inf or betamin == -np.inf:
#                     mask1 = np.ones((n, 1), dtype=np.float32)
#                     mask1[i] = 0
#                     mask1 = tf.Variable(mask1)
#                     mask2 = np.zeros((n, 1), dtype=np.float32)
#                     mask2[i] = beta[i] / 2
#                     mask2 = tf.Variable(mask2)
#                     beta = tf.add(tf.multiply(beta, mask1), mask2)
#                     # beta[i] = beta[i] / 2
#                 else:
#                     mask = np.ones((n, 1), dtype=np.float32)
#                     mask[i] = 0
#                     mask = tf.Variable(mask)
#                     mask2 = np.zeros((n, 1), dtype=np.float32)
#                     mask2[i] = (beta[i] + betamin) / 2
#                     mask2 = tf.Variable(mask2)
#                     beta = tf.add(tf.multiply(beta, mask), mask2)
#                     # beta[i] = (beta[i] + betamin) / 2
#
#             # 更新perb,prob值
#             perp, this_prob = cal_perplexity(dist[i], i, beta[i])
#             perp_diff = perp - base_perp
#             tries = tries + 1
#         # 记录prob值
#         # mask = np.ones((n, d), dtype=np.float32)
#         # mask[i] = 0
#         # mask = tf.Variable(mask)
#         # beta = tf.multiply(beta, mask)
#         pair_prob[i, ] = this_prob
#     print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
#     # 每个点对其他点的条件概率分布pi\j
#     return pair_prob
#
#
# def tsne_loss(x, y, no_dims=2, initial_dims=1024, perplexity=30.0):
#     """Runs t-SNE on the dataset in the NxD array x
#     to reduce its dimensionality to no_dims dimensions.
#     The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
#     where x is an NxD NumPy array.
#     """
#
#     # Check inputs
#     if isinstance(no_dims, float):
#         print("Error: array x should have type float.")
#         return -1
#     if round(no_dims) != no_dims:
#         print("Error: number of dimensions should be an integer.")
#         return -1
#
#     (n, d) = x.shape
#     print(x.shape)
#
#     # 对称化
#     P = seach_prob(x, 1e-5, perplexity)
#     P = P + np.transpose(P)
#     P = P / np.sum(P)  # pij
#     # early exaggeration
#     # pi\j
#     P = np.maximum(P, 1e-12)
#
#     y = tf.cast(y, tf.float32)
#     sum_y = tf.reduce_sum(tf.math.square(y), axis=1)
#     sum_y = tf.cast(sum_y, tf.float32)
#     yT = tf.transpose(y)
#     # num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
#     tmp = tf.math.add(-2 * tf.matmul(y, yT), sum_y)
#     tmp = tf.transpose(tmp)
#     num = 1 / (1 + tf.math.add(tmp, sum_y))
#     diag = tf.linalg.diag(num)
#     num = num - diag
#     # num[range(n), range(n)] = 0
#     # Q = num / np.sum(num)  # qij
#     Q = num / tf.reduce_sum(num)  # qij
#     # Q = np.maximum(Q, 1e-12)  # X与Y逐位比较取其大者
#     Q = tf.math.maximum(Q, 1e-12)  # X与Y逐位比较取其大者
#
#     C = np.sum(P * np.log(P / Q))
#     return C
