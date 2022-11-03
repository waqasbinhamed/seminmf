import numpy as np


def initialize_matrices(m, n, r):
    return np.random.rand(m, r), np.random.rand(r, n)


def non_neg(arr):
    arr[arr < 0] = 0
    return arr


def line_search(c, diff_wis, hj_norm_sq, MhT, _lambda, grad, beta=0.75, itermax=1000, backtracking=False):
    t = 1
    k = 0
    val = func(c, diff_wis, hj_norm_sq, MhT, _lambda)
    # BACKTRACKING DOES NOT WORK
    if backtracking:
        val = val - 0.5 * t * np.linalg.norm(grad) ** 2
    while k < itermax and func(c - t * grad, diff_wis, hj_norm_sq, MhT, _lambda) > val:
        t *= beta
        k += 1
    return t


def func(c, diff_wis_norm, hj_norm_sq, MhT, _lambda):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + _lambda * np.sum(diff_wis_norm)


def update_wj_1(W, Mj, c, hj, j, m, _lambda, itermax=1000):

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

        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[:, norm_mask] = diff_wis[:, norm_mask] / diff_wis_norm[norm_mask]
        tmp[:, ~norm_mask] = unit_norm_vec(m)
        grad = hj_norm_sq * c - MhT + np.sum(tmp, axis=1).reshape(m, 1)

        # simple line search
        t = line_search(c, diff_wis_norm, hj_norm_sq, MhT, _lambda, grad)
        new_c = non_neg(c - t * grad)

        if np.linalg.norm(c - new_c) / np.linalg.norm(c) < 1e-8:
            break
    return new_c


def update_wj_2(W, Mj, new_z, hj, j, m, r, _lambda, rho=0.75, itermax=1000):
    num_edges = r * (r - 1)
    ci_arr = np.delete(W, j, axis=1)
    new_wi_arr = new_yi_arr = np.random.rand(m, r - 1)
    new_yf = new_y0 = np.random.rand(m, 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = non_neg(z - y0 / rho)

        zeta_arr = z - yi_arr / rho
        tmp_arr = zeta_arr / (_lambda + 1e-12) - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - _lambda * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - _lambda * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1).reshape(m, 1) + yf + y0 + np.sum(yi_arr,
                                                                                                             axis=1).reshape(
            m, 1)) / (rho * (2 + num_edges))

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        if np.linalg.norm(z - new_z) / np.linalg.norm(z) < 1e-8:
            break
    return new_z


def update_wj_3(new_wj, hj, Mj, W, j, m, _lambda, mu=1, itermax=1000):
    wi_arr = np.delete(W, j, axis=1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T

    for k in range(itermax):
        wj = new_wj

        tmp_arr = (wj - wi_arr) / mu
        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        tmp_arr[:, norm_mask] = tmp_arr[:, norm_mask] / tmp_norm[norm_mask]

        grad = hj_norm_sq * wj - MhT + np.sum(tmp_arr, axis=1).reshape(m, 1)
        t = line_search(wj, wj - wi_arr, hj_norm_sq, MhT, _lambda, grad)

        new_wj = non_neg(wj - t * grad)

        if np.linalg.norm(wj - new_wj) / np.linalg.norm(wj) < 1e-8:
            break
    return new_wj


def nmf(M, W, H, _lambda=0.0, itermax=1000):
    scores = np.empty((1, itermax))
    scores[:] = np.NaN

    m, n = M.shape
    r = W.shape[1]

    best_score = np.Inf
    W_best = np.empty((m, r))
    H_best = np.empty((r, n))

    Mj = M - W @ H
    for it in range(itermax):
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
        scores[0, it] = np.linalg.norm(M - W @ H, 'fro')
        if scores[0, it] > best_score:
            best_score = scores[0, it]
            W_best = W
            H_best = H
        print(it, scores[0, it])
    return W_best, H_best, scores


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
    W, H, scores = nmf(M, W_ini, H_ini, _lambda=0.4, itermax=1000)
    print('done')
