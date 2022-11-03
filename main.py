# TODO 1: inspect all calculations to confirm they are correct
# TODO 4: make sure parameters are appropriate
# TODO 5: run all 3 wj update methods, test lambda values
import numpy as np


def initialize_matrices(m, n, r):
    return np.random.rand(m, r), np.random.rand(r, n)


def non_neg(arr):
    arr[arr < 0] = 0
    return arr


def line_search(c, diff_wis, hj_norm_sq, MhT, _lambda, search_dir):
    alpha = 1
    while func(c + alpha * search_dir, diff_wis, hj_norm_sq, MhT, _lambda) > func(c, diff_wis, hj_norm_sq, MhT,
                                                                                  _lambda):
        alpha *= 0.5
    return alpha


def func(c, diff_wis_norm, hj_norm_sq, MhT, _lambda):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + _lambda * np.sum(diff_wis_norm)


def backtrack(c, diff_wis, hj_norm_sq, MhT, _lambda, search_dir, beta=0.75):
    # TODO 2: fix backtracking line search
    alpha = 0.3
    t = 1
    while func(c + alpha * search_dir, diff_wis, hj_norm_sq, MhT, _lambda) > func(c, diff_wis, hj_norm_sq, MhT,
                                                                                  _lambda) + alpha * t * (
            -search_dir.T @ search_dir):
        t *= beta
    return t


def update_wj_1(W, Mj, c, hj, j, m, _lambda, itermax=1000):
    def grad_func(MhT, c, diff_wis, diff_wis_norm, hj_norm_sq, m):
        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[:, norm_mask] = diff_wis[:, norm_mask] / diff_wis_norm[norm_mask]
        tmp[:, ~norm_mask] = unit_norm_vec(m)
        grad = hj_norm_sq * c - MhT + np.sum(tmp, axis=1).reshape(m, 1)
        return grad

    def unit_norm_vec(sz):
        tau = np.zeros((sz, 1))
        tau[0] = 1
        return tau

    new_c = c
    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T
    for k in range(itermax):
        c = new_c

        diff_wis = np.delete(W, j, axis=1) - c
        diff_wis_norm = np.linalg.norm(diff_wis, axis=0)
        grad = grad_func(MhT, c, diff_wis, diff_wis_norm, hj_norm_sq, m)

        # simple line search
        t = line_search(c, diff_wis_norm, hj_norm_sq, MhT, _lambda, -grad)
        # t = backtrack(c, diff_wis, hj_norm_sq, MhT, _lambda, -grad)
        new_c = non_neg(c - t * grad)
    return new_c


def update_wj_2(W, Mj, z, hj, j, m, r, _lambda, rho=0.75, itermax=1000):
    num_edges = r * (r - 1)
    yf = y0 = np.random.rand(m, 1)
    ci_arr = np.delete(W, j, axis=1)
    new_wi_arr = yi_arr = np.random.rand(m, r - 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = non_neg(z - y0 / rho)

        zeta_arr = z - (yi_arr / rho)
        tmp_arr = zeta_arr / _lambda - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - _lambda * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - _lambda * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf - new_w0) + rho * np.sum(new_wi_arr, axis=1).reshape(m, 1) + yf + y0 + np.sum(yi_arr,
                                                                                                             axis=1).reshape(
            m, 1)) / (rho * (2 + num_edges))

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr
    return z


def update_wj_3(wj, hj, Mj, W, j, m, _lambda, mu=1, itermax=1000):
    # TODO 3: inspect long runtime
    wi_arr = np.delete(W, j, axis=1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T

    for k in range(itermax):
        tmp_arr = (wj - wi_arr) / mu
        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        tmp_arr[:, norm_mask] = tmp_arr[:, norm_mask] / tmp_norm[norm_mask]

        grad = hj_norm_sq * wj - MhT + np.sum(tmp_arr, axis=1).reshape(m, 1)
        t = line_search(wj, wj - wi_arr, hj_norm_sq, MhT, _lambda, -grad)

        new_wj = non_neg(wj - t * grad)

        wj = new_wj
    return wj


def nmf(M, W, H, _lambda=0.0, iters=100):
    m, n = M.shape
    r = W.shape[1]

    Mj = M - W @ H
    for it in range(iters):
        for j in range(r):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]

            Mj = Mj + wj @ hj

            # update h_j
            H[j: j + 1, :] = hj = non_neg(wj.T @ Mj) / (np.linalg.norm(wj) ** 2)

            # update w_j
            # W[:, j: j + 1] = wj = update_wj_1(W, Mj, wj, hj, j, m, _lambda)
            # W[:, j: j + 1] = wj = update_wj_2(W, Mj, wj, hj, j, m, r, _lambda)
            W[:, j: j + 1] = wj = update_wj_3(wj, hj, Mj, W, j, m, _lambda)
            Mj = Mj - wj @ hj
        print(it, np.linalg.norm(M - W @ H, 'fro'))


if __name__ == '__main__':
    np.random.seed(42)

    def zeros_mask(arr):
        m, n = arr.shape
        indices = np.random.choice(m * n, replace=False, size=int(m * n * 0.4))
        arr[np.unravel_index(indices, (m, n))] = 0
        return arr


    m, n, r_true = 8, 5, 3
    W_true = zeros_mask(np.random.rand(m, r_true))
    H_true = zeros_mask(np.random.rand(r_true, n))
    M = W_true @ H_true

    r = 5
    W_ini = np.random.rand(m, r)
    H_ini = np.random.rand(r, n)
    nmf(M, W_ini, H_ini, _lambda=0.2, iters=20)
    print('done')
