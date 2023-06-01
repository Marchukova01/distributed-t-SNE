import numpy as np
import math


# binary search for a suitable sigma value
def binary_search(X_sqdistances, n_samples, n_steps, P, sum_line_P, begin_line, lines, S, desired_perplexity):
    for i in range(lines):
        sigma_min = -np.NINF
        sigma_max = np.NINF
        EPSILON_DBL = 1e-12
        PERPLEXITY_TOLERANCE = 1e-5

        desired_entropy = math.log(desired_perplexity)
        sigma = 1.0

        # Binary search
        for l in range(n_steps):
            sum_Pi = 0.0
            for j in range(n_samples):
                if j != i + begin_line:
                    P[i, j] = math.exp(-X_sqdistances[i, j] * sigma)
                    sum_Pi += P[i, j]
            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_samples):
                P[i, j] /= sum_Pi
                sum_disti_Pi += X_sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + sum_disti_Pi * sigma
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                sigma_min = sigma
                if sigma_max == np.NINF:
                    sigma *= 2.0
                else:
                    sigma = (sigma + sigma_max) / 2.0
            else:
                sigma_max = sigma
                if sigma_min == -np.NINF:
                    sigma /= 2.0
                else:
                    sigma = (sigma + sigma_min) / 2.0
        S[i] = sigma
        sum_line_P[i + begin_line] = sum_Pi
    return P, sum_line_P


# calculating the squared distance in euclidean metric
def sqdistances(X_sqdistances, X, n_samples, m_coord, lines, begin_line):
    for i in range(lines):
        for j  in range(n_samples):
            if (j != i + begin_line):
                for k in range(m_coord):
                    X_sqdistances[i, j] += (X[k, i + begin_line] - X[k, j])**2
    return X_sqdistances


#calculating the derivative of the Kullback-Leibler divergence
def gradient_d(i, n, line_num, PQ, Y, dim_low, A, dY):
    nowY = np.zeros((dim_low, n), dtype=float)
    for j in range(n):
      nowY[:, j] = Y[:, i] - Y[:, j]
    dY[i, :] = np.sum(np.tile(PQ[line_num, :] * A[line_num, :], (dim_low, 1)) * (nowY),  1)
    return dY

