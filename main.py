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
            W[:, j: j + 1] = wj = update_wj_1(W, Mj, wj, hj, j, m, _lambda)

            Mj = Mj - wj @ hj


def update_wj_2(W, Mj, z, hj, j, m, r, _lambda, rho=0.75, itermax=100):
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
        # TODO: fix this
        new_wi_arr[:, norm_mask] = zeta_arr - _lambda * (tmp_arr / tmp_norm)
        new_wi_arr[:, ~norm_mask] = zeta_arr - _lambda * tmp_arr

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


def non_neg(arr):
    arr[arr < 0] = 0
    return arr


def update_wj_1(W, Mj, c, hj, j, m, _lambda, itermax=100):
    def grad_func(MhT, c, diff_wis, hj_norm_sq, m):
        diff_wis_norm = np.linalg.norm(diff_wis, axis=0)
        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[: norm_mask] = diff_wis[: norm_mask] / diff_wis_norm[norm_mask]
        tmp[: ~norm_mask] = unit_norm_vec(m)
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
        grad = grad_func(MhT, c, diff_wis, hj_norm_sq, m)

        # simple line search
        t = line_search(c, diff_wis, hj_norm_sq, MhT, _lambda, -grad)
        new_c = non_neg(c - t * grad)
    return new_c


def update_wj_3(wj, W, j, m, r, MhT, hj_norm_sq, _lambda, mu=1, itermax=100):
    wi_arr = np.delete(W, j, axis=1)

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



def line_search(c, diff_wis, hj_norm_sq, MhT, _lambda, search_dir):
    alpha = 1
    while func(c + alpha * search_dir, diff_wis, hj_norm_sq, MhT, _lambda) > func(c, diff_wis, hj_norm_sq, MhT,
                                                                                  _lambda):
        alpha *= 0.5
    return alpha


def func(c, diff_wis, hj_norm_sq, MhT, _lambda):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + _lambda * np.sum(diff_wis, axis=1)


# def backtrack(og_func, grad, wj, hj_norm_sq, MhT, _lambda, W, j, beta=0.4):
#     alpha = 0.3
#     t = 1
#     # search direction = -gradient
#     while func(wj - t*grad, hj_norm_sq, MhT, _lambda, W, j) > og_func - alpha * t * (grad.T @ grad):
#         t *= beta
#         print(t)
#     return t

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
