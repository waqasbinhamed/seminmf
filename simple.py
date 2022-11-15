import numpy as np


def non_neg(arr):
    arr[arr < 0] = 0
    return arr


def calculate_gscore(W, r):
    gscore = 0
    for i in range(r):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def update_wj(W, Mj, new_z, hj, j, m, r, _lambda, rho=0.75, itermax=1000):
    num_edges = r * (r - 1)
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    new_yi_arr = np.random.rand(m, r - 1)
    new_yf = np.random.rand(m, 1)
    new_y0 = np.random.rand(m, 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = non_neg(z - y0 / rho)

        zeta_arr = z - yi_arr / rho
        tmp_arr = zeta_arr / _lambda - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - _lambda * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - _lambda * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0 + np.sum(yi_arr,
                                                                                                              axis=1,
                                                                                                              keepdims=True)) / (
                            rho * (2 + num_edges))

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def nmf_son(M, W, H, _lambda=0.0, itermax=1000, scale_lambda=False, verbose=False):
    m, n = M.shape
    r = W.shape[1]

    fscores = np.zeros((itermax + 1,))
    gscores = np.zeros((itermax + 1,))
    lambda_vals = np.zeros((itermax + 1,))

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W, r)

    if scale_lambda:
        scaled_lambda = lambda_vals[0] = (fscores[0] / gscores[0]) * _lambda
    else:
        scaled_lambda = _lambda
        lambda_vals[:] = _lambda

    best_score = np.Inf
    W_best = np.zeros((m, r))
    H_best = np.zeros((r, n))

    Mj = M - W @ H
    for it in range(1, itermax + 1):
        for j in range(r):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]

            Mj = Mj + wj @ hj

            # update h_j
            H[j: j + 1, :] = hj = non_neg(wj.T @ Mj) / (np.linalg.norm(wj) ** 2)

            # update w_j
            W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, m, r, scaled_lambda)

            Mj = Mj - wj @ hj

        fscores[it] = np.linalg.norm(M - W @ H, 'fro')
        gscores[it] = calculate_gscore(W, r)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        if scale_lambda:
            scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores, gscores, lambda_vals
