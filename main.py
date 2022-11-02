import numpy as np


def initialize_matrices(m, n, r):
    return np.random.rand(m, r), np.random.rand(r, n)


def nmf(M, W, H, _lambda=0, iters=100):
    m, n = M.shape
    r = W.shape[1]

    Mj = M - W @ H
    for it in range(iters):
        for j in range(r):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]

            Mj = Mj + wj @ hj

            # update h_j
            H[j: j + 1, :] = hj = update_hj(Mj, wj)

            # TODO: Implement Algorithm 3
            W[:, j: j + 1] = wj = update_wj_2(W, Mj, wj, hj, j, m, r, _lambda, rho=1, itermax=100)

            Mj = Mj - wj @ hj


def update_wj_2(W, Mj, z, hj, j, m, r, _lambda, rho=1, itermax=100):
    num_edges = r * (r-1)
    yf = y0 = np.random.rand(m, 1)
    ci_arr = np.delete(W, j, axis=1)
    new_wi_arr = new_yi_arr = yi_arr = np.random.rand(m, r-1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        new_wf = (Mj @ hj.T - yf + rho*z) / (rho + hj_norm_sq)

        tmp1 = z - (y0 / rho)
        tmp1[tmp1 < 0] = 0
        new_w0 = tmp1

        # try to parallelize
        for i in range(r-1):
            zeta = z - (yi_arr[:, i: i+1] / rho)
            tmp2 = (zeta / _lambda) - ci_arr[:, i: i+1]
            tmp2_norm = np.linalg.norm(tmp2)
            if tmp2_norm > 1:
                new_wi_arr[:, i: i+1] = zeta - _lambda * (tmp2 / tmp2_norm)
            else:
                new_wi_arr[:, i: i+1] = zeta - _lambda * tmp2

        new_z = (rho*(new_wf - new_w0) + rho * np.sum(new_wi_arr, axis=1).reshape(m, 1) + yf + y0 + np.sum(yi_arr, axis=1).reshape(m, 1)) / (rho * (2 + num_edges))

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        for i in range(r-1):
            new_yi_arr[:, i: i+1] = yi_arr[:, i: i+1] + rho * (new_wi_arr[:, i: i+1] - new_z)

        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        # TODO: penalty update
        rho = 1.3 * rho
    return z


def update_wj_1(W, Mj, wj, hj, j, m, _lambda, beta=0.7):
    k = 0
    new_wj = wj

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T
    while (np.linalg.norm(new_wj - wj) > 1e-4 and k < 100) or k == 0:
        wj = new_wj
        diffW = np.delete(W, j, axis=1) - wj
        diffW_norm = np.linalg.norm(diffW, axis=0)
        grad = hj_norm_sq * wj - MhT + np.sum(diffW / diffW_norm, axis=1).reshape(m, 1)
        og_func = 0.5 * hj_norm_sq * (np.linalg.norm(wj) ** 2) - np.dot(MhT.T, wj) + _lambda * np.sum(diffW_norm)

        # backtracking line search
        t = backtrack(og_func, grad, wj, hj_norm_sq, MhT, _lambda, W, j, beta)

        tmp2 = wj - t * grad
        tmp2[tmp2 < 0] = 0
        new_wj = tmp2
        k += 1
    return new_wj


def func(wj, hj_norm_sq, MhT, _lambda, W, j):
    diffW = np.delete(W, j, axis=1) - wj
    diffW_norm = np.linalg.norm(diffW, axis=0)
    return 0.5 * hj_norm_sq * (np.linalg.norm(wj) ** 2) - np.dot(MhT.T, wj) + _lambda * np.sum(diffW_norm)


def backtrack(og_func, grad, wj, hj_norm_sq, MhT, _lambda, W, j, beta=0.4):
    alpha = 0.3
    t = 1
    # search direction = -gradient
    while func(wj - t*grad, hj_norm_sq, MhT, _lambda, W, j) > og_func - alpha * t * (grad.T @ grad):
        t *= beta
        print(t)
    return t

def update_hj(Mj, wj):
    tmp = wj.T @ Mj
    tmp[tmp < 0] = 0
    new_hj = tmp / (np.linalg.norm(wj) ** 2)
    return new_hj


if __name__ == '__main__':
    m, n, r = 20, 10, 5
    M0 = np.random.rand(m, n)
    W0, H0 = initialize_matrices(m, n, r)
    nmf(M0, W0, H0)
