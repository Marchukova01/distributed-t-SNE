import numpy as np

import utils
import calc_probability


def tsne(X, Y, y, n_samples, m_coord, dim_low, d, max_iter, n_steps, early_exaggeration, desired_perplexity, eta):
    initial_momentum = 0.5
    final_momentum = 0.8
    min_gain = 0.01

    dY = np.zeros((n_samples, dim_low))
    iY = np.zeros((n_samples, dim_low))
    gains = np.ones((n_samples, dim_low))

    begin_line0 = 0
    last_line0 = int(n_samples / d - 1)
    lines0 = last_line0 - begin_line0 + 1

    begin_line1 = int(n_samples / d)
    last_line1 = int(n_samples * 2 / d - 1)
    lines1 = last_line1 - begin_line1 + 1

    begin_line2 = int(n_samples * 2 / d)
    last_line2 = int(n_samples * 3 / d - 1)
    lines2 = last_line2 - begin_line2 + 1

    begin_line3 = int(n_samples * 3 / d)
    last_line3 = int(n_samples * 4 / d - 1)
    lines3 = last_line3 - begin_line3 + 1

    line_num0 = 0
    line_num1 = 0
    line_num2 = 0
    line_num3 = 0

    S0 = np.zeros(lines0)
    S1 = np.zeros(lines1)
    S2 = np.zeros(lines2)
    S3 = np.zeros(lines3)

    S = np.zeros(n_samples)

    sumQ = np.zeros(n_samples)

    initial_P0 = np.zeros((last_line0 - begin_line0 + 1, n_samples))
    initial_P1 = np.zeros((last_line1 - begin_line1 + 1, n_samples))
    initial_P2 = np.zeros((last_line2 - begin_line2 + 1, n_samples))
    initial_P3 = np.zeros((last_line3 - begin_line3 + 1, n_samples))

    result_P0 = np.zeros((last_line0 - begin_line0 + 1, n_samples))
    result_P1 = np.zeros((last_line1 - begin_line1 + 1, n_samples))
    result_P2 = np.zeros((last_line2 - begin_line2 + 1, n_samples))
    result_P3 = np.zeros((last_line3 - begin_line3 + 1, n_samples))

    sum_Pi = np.zeros(n_samples)

    Q0 = np.zeros((last_line0 - begin_line0 + 1, n_samples))
    Q1 = np.zeros((last_line1 - begin_line1 + 1, n_samples))
    Q2 = np.zeros((last_line2 - begin_line2 + 1, n_samples))
    Q3 = np.zeros((last_line3 - begin_line3 + 1, n_samples))

    A0 = np.zeros((last_line0 - begin_line0 + 1, n_samples))
    A1 = np.zeros((last_line1 - begin_line1 + 1, n_samples))
    A2 = np.zeros((last_line2 - begin_line2 + 1, n_samples))
    A3 = np.zeros((last_line3 - begin_line3 + 1, n_samples))

    X_sqdistances0 = np.zeros((lines0, n_samples))
    X_sqdistances1 = np.zeros((lines1, n_samples))
    X_sqdistances2 = np.zeros((lines2, n_samples))
    X_sqdistances3 = np.zeros((lines3, n_samples))

    Y_sqdistances0 = np.zeros((lines0, n_samples))
    Y_sqdistances1 = np.zeros((lines1, n_samples))
    Y_sqdistances2 = np.zeros((lines2, n_samples))
    Y_sqdistances3 = np.zeros((lines3, n_samples))

    sum_X = np.sum(np.square(X), axis=0)

    numX0 = -2. * np.dot(X[:, begin_line0:last_line0 + 1].T, X)
    numX1 = -2. * np.dot(X[:, begin_line1:last_line1 + 1].T, X)
    numX2 = -2. * np.dot(X[:, begin_line2:last_line2 + 1].T, X)
    numX3 = -2. * np.dot(X[:, begin_line3:last_line3 + 1].T, X)

    X_sqdistances0 = np.add(np.add(numX0, sum_X).T, sum_X[begin_line0:last_line0 + 1]).T
    X_sqdistances1 = np.add(np.add(numX1, sum_X).T, sum_X[begin_line1:last_line1 + 1]).T
    X_sqdistances2 = np.add(np.add(numX2, sum_X).T, sum_X[begin_line2:last_line2 + 1]).T
    X_sqdistances3 = np.add(np.add(numX3, sum_X).T, sum_X[begin_line3:last_line3 + 1]).T

    initial_P0, sum_Pi = utils.binary_search(X_sqdistances0, n_samples, n_steps, initial_P0, sum_Pi, begin_line0, lines0, S0,
                                       desired_perplexity)
    initial_P1, sum_Pi = utils.binary_search(X_sqdistances1, n_samples, n_steps, initial_P1, sum_Pi, begin_line1, lines1, S1,
                                       desired_perplexity)
    initial_P2, sum_Pi = utils.binary_search(X_sqdistances2, n_samples, n_steps, initial_P2, sum_Pi, begin_line2, lines2, S2,
                                       desired_perplexity)
    initial_P3, sum_Pi = utils.binary_search(X_sqdistances3, n_samples, n_steps, initial_P3, sum_Pi, begin_line3, lines3, S3,
                                       desired_perplexity)

    sum_Pi = np.maximum(sum_Pi, 1e-12)

    S = np.hstack([S0, S1, S2, S3])

    result_P0 = calc_probability.calcP(n_samples, m_coord, X, initial_P0, result_P0, sum_Pi, begin_line0, lines0, S, early_exaggeration)
    result_P1 = calc_probability.calcP(n_samples, m_coord, X, initial_P1, result_P1, sum_Pi, begin_line1, lines1, S, early_exaggeration)
    result_P2 = calc_probability.calcP(n_samples, m_coord, X, initial_P2, result_P2, sum_Pi, begin_line2, lines2, S, early_exaggeration)
    result_P3 = calc_probability.calcP(n_samples, m_coord, X, initial_P3, result_P3, sum_Pi, begin_line3, lines3, S, early_exaggeration)

    for iter in range(max_iter):

        Q0 = np.full((last_line0 - begin_line0 + 1, n_samples), 0)
        Q1 = np.full((last_line1 - begin_line1 + 1, n_samples), 0)
        Q2 = np.full((last_line2 - begin_line2 + 1, n_samples), 0)
        Q3 = np.full((last_line3 - begin_line3 + 1, n_samples), 0)

        line_num0 = 0
        line_num1 = 0
        line_num2 = 0
        line_num3 = 0

        sum_Y = np.sum(np.square(Y), axis=0)
        num0 = -2. * np.dot(Y[:, begin_line0:last_line0 + 1].T, Y)
        num1 = -2. * np.dot(Y[:, begin_line1:last_line1 + 1].T, Y)
        num2 = -2. * np.dot(Y[:, begin_line2:last_line2 + 1].T, Y)
        num3 = -2. * np.dot(Y[:, begin_line3:last_line3 + 1].T, Y)

        Y_sqdistances0 = np.add(np.add(num0, sum_Y).T, sum_Y[begin_line0:last_line0 + 1]).T
        Y_sqdistances1 = np.add(np.add(num1, sum_Y).T, sum_Y[begin_line1:last_line1 + 1]).T
        Y_sqdistances2 = np.add(np.add(num2, sum_Y).T, sum_Y[begin_line2:last_line2 + 1]).T
        Y_sqdistances3 = np.add(np.add(num3, sum_Y).T, sum_Y[begin_line3:last_line3 + 1]).T

        Q0, sumQ = calc_probability.calcQ(n_samples, Q0, Y_sqdistances0, begin_line0, lines0, sumQ)
        Q1, sumQ = calc_probability.calcQ(n_samples, Q1, Y_sqdistances1, begin_line1, lines1, sumQ)
        Q2, sumQ = calc_probability.calcQ(n_samples, Q2, Y_sqdistances2, begin_line2, lines2, sumQ)
        Q3, sumQ = calc_probability.calcQ(n_samples, Q3, Y_sqdistances3, begin_line3, lines3, sumQ)

        sum = np.sum(sumQ)

        A0 = Q0
        A1 = Q1
        A2 = Q2
        A3 = Q3

        Q0 = Q0 / sum
        Q1 = Q1 / sum
        Q2 = Q2 / sum
        Q3 = Q3 / sum

        Q0 = np.maximum(Q0, 1e-12)
        Q1 = np.maximum(Q1, 1e-12)
        Q2 = np.maximum(Q2, 1e-12)
        Q3 = np.maximum(Q3, 1e-12)

        # Compute gradient
        PQ0 = result_P0 - Q0
        PQ1 = result_P1 - Q1
        PQ2 = result_P2 - Q2
        PQ3 = result_P3 - Q3

        for i in range(n_samples):
            if (i < begin_line1):
                dY = utils.gradient_d(i, n_samples, line_num0, PQ0, Y, dim_low, A0, dY)
                line_num0 += 1
            elif (i < begin_line2):
                dY = utils.gradient_d(i, n_samples, line_num1, PQ1, Y, dim_low, A1, dY)
                line_num1 += 1
            elif (i < begin_line3):
                dY = utils.gradient_d(i, n_samples, line_num2, PQ2, Y, dim_low, A2, dY)
                line_num2 += 1
            else:
                dY = utils.gradient_d(i, n_samples, line_num3, PQ3, Y, dim_low, A3, dY)
                line_num3 += 1

        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum


        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain

        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY.T
        Y = Y - np.tile(np.mean(Y, 1), (n_samples, 1)).T

        if iter == 100:
            result_P0 = result_P0 / early_exaggeration
            result_P1 = result_P1 / early_exaggeration
            result_P2 = result_P2 / early_exaggeration
            result_P3 = result_P3 / early_exaggeration

    return Y

