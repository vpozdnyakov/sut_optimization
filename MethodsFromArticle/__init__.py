import numpy as np
import numpy.linalg as alg
import math
import copy


#  RAS METHOD
def diog(x):
    return np.eye(x.shape[0]) * x


def RAS_pred(previous_year, u, v, eps):
    A = previous_year
    A = A + 10 ** (-8)  # We need to add some noise in order to avoid mistakes computing reverse matrix
    r = np.ones(A.shape[0])
    s = np.ones(A.shape[1])
    while 1:
        matrix1 = diog((A.dot(alg.inv(diog(A.T.dot(r))))).dot(v))
        matrix2 = diog((A.T.dot(alg.inv(diog(A.dot(s))))).dot(u))
        new_r = alg.inv(matrix1).dot(u)
        new_s = alg.inv(matrix2).dot(v)
        if alg.norm(new_r - r) < eps and alg.norm(new_s - s) < eps:
            r = (alg.inv(diog(A.dot(new_s)))).dot(u)
            s = (alg.inv(diog(A.T.dot(new_r)))).dot(v)
            break
        r = new_r
        s = new_s
    return (diog(r).dot(A)).dot(diog(s))


# Improved Normalized squared difference (INS)
def next_lambda(lambd, tau, z, A, u, M):
    min_z = np.where(z < 0, z, 0)
    result = lambd.copy()
    for i in range(result.shape[0]):
        expr = A[i] * min_z[i] * M - tau * np.abs(A[i])
        denominator = np.sum(np.abs(A[i]))
        if denominator == 0:
            result[i] = 0
        else:
            result[i] = (u[i] - np.sum(A[i]) + np.sum(expr)) / denominator
    return result


def next_tau(lambd, tau, z, A, v, M):
    min_z = np.where(z.T < 0, z.T, 0)
    result = tau.copy()
    for j in range(result.shape[0]):
        expr = A.T[j] * min_z[j] * M - lambd * np.abs(A.T[j])
        denominator = np.sum(np.abs(A.T[j]))
        if denominator == 0:
            result[j] = 0
        else:
            result[j] = (v[j] - np.sum(A.T[j]) + np.sum(expr)) / denominator
    return result


def compute_z(lambd, tau, z, A, M):
    result = copy.deepcopy(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if A[i][j] == 0:
                result[i][j] = 1
            else:
                var = 1 + ((lambd[i] + tau[j]) * A[i][j] / abs(A[i][j]))
                if var >= 0:
                    result[i][j] = var
                else:
                    result[i][j] = var / (1 + M)
    return result


#  Improved square differences (ISD)
def ISDnext_lambda(lambd, tau, z, A, u, M):
    min_z = np.where(z < 0, z, 0)
    result = lambd.copy()
    delta_A = np.where(A != 0, 1, 0)
    for i in range(result.shape[0]):
        expr = A[i] * min_z[i] * M - tau * delta_A[i]
        denominator = np.sum(delta_A[i])
        if denominator == 0:
            result[i] = 0
        else:
            result[i] = (u[i] - np.sum(A[i]) + np.sum(expr)) / denominator
    return result


def ISDnext_tau(lambd, tau, z, A, v, M):
    min_z = np.where(z.T < 0, z.T, 0)
    result = tau.copy()
    delt_A = np.where(A != 0, 1, 0)
    for j in range(result.shape[0]):
        expr = A.T[j] * min_z[j] * M - lambd * delt_A.T[j]
        denominator = np.sum(delt_A.T[j])
        if denominator == 0:
            result[j] = 0
        else:
            result[j] = (v[j] - np.sum(A.T[j]) + np.sum(expr)) / denominator
    return result


def ISDcompute_z(lambd, tau, z, A, M):
    result = copy.deepcopy(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if A[i][j] == 0:
                result[i][j] = 1
            else:
                var = 1 + ((lambd[i] + tau[j]) / A[i][j])
                if var >= 0:
                    result[i][j] = var
                else:
                    result[i][j] = var / (1 + M)
    return result


#  Improved weighted square differences (IWS)
def IWSnext_lambda(lambd, tau, z, A, u, M, devided_A):
    min_z = np.where(z < 0, z, 0)
    result = lambd.copy()
    for i in range(result.shape[0]):
        expr = A[i] * min_z[i] * M - tau * devided_A[i]
        denominator = np.sum(devided_A[i])
        if denominator == 0:
            result[i] = 0
        else:
            result[i] = (u[i] - np.sum(A[i]) + np.sum(expr)) / denominator
    return result


def IWSnext_tau(lambd, tau, z, A, v, M, devided_A):
    min_z = np.where(z.T < 0, z.T, 0)
    result = tau.copy()
    for j in range(result.shape[0]):
        expr = A.T[j] * min_z[j] * M - lambd * devided_A.T[j]
        denominator = np.sum(devided_A.T[j])
        if denominator == 0:
            result[j] = 0
        else:
            result[j] = (v[j] - np.sum(A.T[j]) + np.sum(expr)) / denominator
    return result


def IWScompute_z(lambd, tau, z, A, M):
    result = copy.deepcopy(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if A[i][j] == 0:
                result[i][j] = 1
            else:
                var = 1 + ((lambd[i] + tau[j]) * A[i][j]) / (abs(A[i][j]) ** 3)
                if var >= 0:
                    result[i][j] = var
                else:
                    result[i][j] = var / (1 + M)
    return result


methods = {'INS': [next_lambda, next_tau, compute_z], 'ISD': [ISDnext_lambda, ISDnext_tau, ISDcompute_z],
           'IWS': [IWSnext_lambda, IWSnext_tau, IWScompute_z]}


def predict(previous_year, u, v, method='INS', M=100, eps=1e-8):
    '''This is the main function which taken the previous year data can predict matrix for the next year.
    variables:
    previous_year:
        data of the previous year
    u, v:
        sum by rows and columns respectively
    method:
        It is one of the following strings: "INS", "ISD", "IWS". The chosen method will be used
    M:
        refer to the hyperparameter. For more details refer to the "MainArticle" which is:
        Projection of Supply and Use tables: methods and their empirical assessment
        Working Paper Number: 2
        Authors: Umed Temurshoev, Norihiko Yamano and Colin Webb'''
    if method == 'RAS':
        return RAS_pred(previous_year, u, v, eps)
    A = previous_year
    z = np.ones(A.shape)
    lambd = np.zeros(u.shape[0])
    tau = np.zeros(v.shape[0])

    if method == 'IWS':
        devided_A = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                devided_A[i][j] = 1 / A[i][j] if A[i][j] != 0 else 0

        while 1:
            new_lambd = methods[method][0](lambd, tau, z, A, u, M, devided_A)
            new_tau = methods[method][1](new_lambd, tau, z, A, v, M, devided_A)
            z = methods[method][2](new_lambd, new_tau, z, A, M)
            if np.linalg.norm(new_lambd - lambd) < eps and np.linalg.norm(new_tau - tau) < eps:
                break
            lambd = new_lambd
            tau = new_tau
        pred_current_year = z * A
        return pred_current_year

    while 1:
        new_lambd = methods[method][0](lambd, tau, z, A, u, M)
        new_tau = methods[method][1](new_lambd, tau, z, A, v, M)
        z = methods[method][2](new_lambd, new_tau, z, A, M)
        if np.linalg.norm(new_lambd - lambd) < eps and np.linalg.norm(new_tau - tau) < eps:
            break
        lambd = new_lambd
        tau = new_tau
    pred_current_year = z * A
    return pred_current_year


# GradProjection Method
def find_actives(used, X):
    '''This function is ment to find all the active restrictions'''
    activ = []
    for i in range(X.shape[0]):
        if i in used:
            continue
        if abs(-X[i] - 1) < 1e-10:
            activ.append(i)
    return set(activ)


def predict_grad(previous_year, u, v, eps=1e-8):
    D = previous_year

    m = D.shape[1]
    n = D.shape[0]

    C = np.eye(D.shape[0] * D.shape[1]) * D.flatten()  # Quadratic matrix in transition from matrices to vectors
    A = np.zeros((D.shape[0] + D.shape[1], D.shape[0] * D.shape[1]))  # Restriction matrix Ax  = b
    for i in range(n):
        for j in range(i * m, (i + 1) * m):
            A[i][j] = C[j][j]
    for i in range(n, n + m):
        for j in range(i - n, n * m, m):
            A[i][j] = C[j][j]

    M = np.eye(n * m, n * m) * -1
    b = np.array([1] * (n * m))
    used = set()  # A set of previously considred active restrictions.

    u_v = np.concatenate((u, v), axis=0)
    new_u_v = u_v - A.dot(
        np.ones(A.shape[1]))  # Vectors union (и and v) and modifications of our restrictions into Ax = u_v
    X = alg.pinv(A).dot(new_u_v)  # Initial approximation

    actives = find_actives(used, X)
    for i in actives:
        A = np.concatenate((A, [M[i]]), axis=0)
    used = used.union(actives)

    P = np.eye(A.shape[1]) - alg.pinv(A).dot(A)  # Projection matrix

    a_abs = np.abs(C.diagonal())  # A matrix of a square form at transition from matrices to vectors
    # (to be more precise, its diagonal, we should not forget that in
    #  our function these are modules |a|).
    f_val = np.sum(a_abs * X ** 2)  # Target function value under initial approximation.
    while 1:

        d = (-1) * P.dot(2 * a_abs * X)  # Antigradient direction, projected to the permissible range
        if alg.norm(d) < 1e-8:
            break

        T = 10 ** 20  # We want to make T as big as possible to avoid any mistakes
        for i in range(M.shape[0]):
            if i in used:
                continue
            if M[i].dot(d) != 0:
                new_T = (b[i] - M[i].dot(X)) / M[i].dot(d)
                T = min(T, new_T)

        T = max(0, T)
        t = - d.dot(C.dot(X)) / d.dot(C.dot(d))  # Optimal stride value
        t = min(t, T)

        new_X = X + t * d
        new_f = np.sum(a_abs * new_X ** 2)
        if abs((f_val / new_f) - 1) < eps:
            X = new_X
            break
        X = new_X
        f_val = new_f

        actives = find_actives(used, X)
        for i in actives:
            A = np.concatenate((A, [M[i]]), axis=0)
        used = used.union(actives)

        P = np.eye(A.shape[1]) - alg.pinv(A).dot(A)  # Projection matrix refreshment

    return ((X + 1) * C.diagonal()).reshape(len(u), len(v))
