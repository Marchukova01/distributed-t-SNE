import numpy as np
import math


# calculating a probability value stored on another machine
def help_P_ji(i, j, X, m_coord, P, sum_Pi, S):
    X_sqdistances = 0.0
    for r in range(m_coord):
        X_sqdistances += (X[r, i] - X[r, j]) ** 2
    p = math.exp(-X_sqdistances * S[j])
    p /= sum_Pi[j]
    return p


# calculation of the probability of proximity of points in the original space
def calcP(n_samples, m_coord, X, initial_P, result_P, sum_Pi, begin_line, lines, S, early_exaggeration):
    for i in range(lines):
        for j in range(0, n_samples):
            if (j != i + begin_line):
                P_ji = help_P_ji(i + begin_line, j, X, m_coord, initial_P, sum_Pi, S)
                result_P[i, j] = (initial_P[i, j] + P_ji)/(2 * n_samples)
    result_P = result_P * early_exaggeration
    result_P = np.maximum(result_P, 1e-12)
    return result_P


# calculation of the proximity probability of points in the embedding space
def calcQ(n_samples, Q, Y_sqdistances, begin_line, lines, sum):
    EPSILON_DBL = 1e-12
    for i in range(lines):
        sum[i + begin_line] = 0.0
        for j in range(n_samples):
            if j != i + begin_line:
                Q[i, j] = 1 + Y_sqdistances[i, j]
                Q[i, j] = 1.0/Q[i, j]
                sum[i + begin_line] += Q[i, j]
        if sum[i + begin_line] == 0.0:
            sum[i + begin_line] = EPSILON_DBL
    return Q, sum
