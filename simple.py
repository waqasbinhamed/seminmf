import numpy as np

TOL = 1e-12


def non_neg(arr):
    """Returns non-negative projection of array."""
    arr[arr < 0] = 0
    return arr


def calculate_gscore(W):
    """Calculates the sum of norm of the W matrix."""
    rank = W.shape[1]
    gscore = 0
    for i in range(rank - 1):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def find_andersen_coeff(R_history):
    """
    Compute the combination coefficients alpha_i in Anderson acceleration, i.e., solve
        argmin sum_{i=0}^m alpha_i r_{k-i},  s.t. sum_{i=0}^{m} alpha_i = 1
    Solve using the equivalent least square problem by eliminating the constraint
    """
    nc = R_history.shape[1]

    # Construct least square matrix
    if nc == 1:
        c = np.ones(1)
    else:
        Y = R_history[:, 1:] - R_history[:, 0:-1]
        b = R_history[:, -1]
        q, r = np.linalg.qr(Y)

        z = np.linalg.solve(r, q.T @ b)
        c = np.r_[z[0], z[1:] - z[0:-1], 1 - z[-1]]

    return c


def update_wj(W, Mj, new_z, hj, j, _lambda, itermax=1000):
    """Calculates the w_j vector without acceleration."""
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
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

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < TOL:
            break

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def update_wj_andersen_z(W, Mj, new_z, hj, j, _lambda, itermax=1000, andersen_win=None):
    """Calculates the w_j vector with andersen acceleration of only the dual vector z."""
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
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

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < TOL:
            break

        if andersen_win is not None:
            if it > 1:
                prev_zs = np.c_[prev_zs, new_z]
                z_residuals = np.c_[z_residuals, new_z - z]
                if z_residuals.shape[1] > andersen_win:
                    prev_zs = prev_zs[:, 1:]
                    z_residuals = z_residuals[:, 1:]

                cz = find_andersen_coeff(z_residuals)
                new_z = (prev_zs @ cz).reshape(m, 1)

            else:
                prev_zs = new_z.copy()
                z_residuals = new_z - z

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def update_wj_andersen_all(W, Mj, new_z, hj, j, _lambda, itermax=1000, andersen_win=None):
    """Calculates the w_j vector with andersen acceleration of w0, wf, wis, and z."""
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
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

        if andersen_win and it > 1:
            prev_wfs = np.c_[prev_wfs, new_wf]
            wf_residuals = np.c_[wf_residuals, new_wf - z]
            if wf_residuals.shape[1] > andersen_win:
                prev_wfs = prev_wfs[:, 1:]
                wf_residuals = wf_residuals[:, 1:]

            cwf = find_andersen_coeff(wf_residuals)
            new_wf = (prev_wfs @ cwf).reshape(m, 1)

            prev_w0s = np.c_[prev_w0s, new_w0]
            w0_residuals = np.c_[w0_residuals, new_w0 - z]
            if w0_residuals.shape[1] > andersen_win:
                prev_w0s = prev_w0s[:, 1:]
                w0_residuals = w0_residuals[:, 1:]

            cw0 = find_andersen_coeff(w0_residuals)
            new_w0 = (prev_w0s @ cw0).reshape(m, 1)
            
            prev_wis_arr = np.dstack((prev_wis_arr, new_wi_arr))
            wi_residuals_arr = np.dstack((wi_residuals_arr, new_wi_arr - z))
            if wi_residuals_arr.shape[2] > andersen_win:
                prev_wis_arr = prev_wis_arr[:, :, 1:]
                wi_residuals_arr = wi_residuals_arr[:, :, 1:]

            for i in range(r-1):
                cwi = find_andersen_coeff(wi_residuals_arr[:, i, :])
                new_wi_arr[:, i: i+1] = (prev_wis_arr[:, i, :] @ cwi).reshape(m, 1)

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < TOL:
            break

        if andersen_win:
            if it > 1:
                prev_zs = np.c_[prev_zs, new_z]
                z_residuals = np.c_[z_residuals, new_z - z]
                if z_residuals.shape[1] > andersen_win:
                    prev_zs = prev_zs[:, 1:]
                    z_residuals = z_residuals[:, 1:]

                cz = find_andersen_coeff(z_residuals)
                new_z = (prev_zs @ cz).reshape(m, 1)

            else:
                prev_zs = new_z.copy()
                z_residuals = new_z - z

                prev_wfs = new_wf.copy()
                wf_residuals = new_wf - z
                
                prev_w0s = new_wf.copy()
                w0_residuals = new_w0 - z

                prev_wis_arr = new_wi_arr.copy()
                wi_residuals_arr = new_wi_arr - z

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def sep_update_H(M, W, H):
    """Calculates the updated H without altering the W matrix."""
    Mj = M - W @ H
    for j in range(W.shape[1]):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        H[j: j + 1, :] = hj = non_neg(wj.T @ Mj) / (np.linalg.norm(wj) ** 2)
        Mj = Mj - wj @ hj

    return H


def sep_update_W(M, W, H, scaled_lambda):
    """Calculates the updated W without altering the H matrix."""
    m, rank = W.shape
    Mj = M - W @ H
    for j in range(rank):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, m, rank, scaled_lambda)
        Mj = Mj - wj @ hj

    return W


def nmf_son(M, W, H, _lambda=0.0, itermax=1000, andersen_type=None, andersen_win=None, verbose=False):
    """Calculates NMF decomposition of the M matrix with andersen acceleration options."""
    m, n = M.shape
    r = W.shape[1]

    fscores = np.zeros((itermax + 1,))
    gscores = np.zeros((itermax + 1,))
    lambda_vals = np.zeros((itermax + 1,))

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W, r)

    scaled_lambda = lambda_vals[0] = (fscores[0] / gscores[0]) * _lambda

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
            W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, scaled_lambda)
            if andersen_type and andersen_win:
                if andersen_type == 'z':
                    W[:, j: j + 1] = wj = update_wj_andersen_z(W, Mj, wj, hj, j, scaled_lambda,
                                                               andersen_win=andersen_win)
                elif andersen_type == 'all':
                    W[:, j: j + 1] = wj = update_wj_andersen_all(W, Mj, wj, hj, j, scaled_lambda,
                                                                 andersen_win=andersen_win)
            else:
                W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, scaled_lambda)

            Mj = Mj - wj @ hj

        fscores[it] = np.linalg.norm(M - W @ H, 'fro')
        gscores[it] = calculate_gscore(W, r)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores, gscores, np.r_[np.NaN, lambda_vals[1:]]


def nmf_son_acc(M, W, H, _lambda=0.0, itermax=1000, verbose=False):
    """Calculates NMF decomposition of the M matrix with new acceleration."""
    beta, _beta, gr, _gr, decay = 0.5, 1, 1.05, 1.01, 1.5

    m, n = M.shape
    rank = W.shape[1]

    fscores = np.zeros((itermax + 1,))
    gscores = np.zeros((itermax + 1,))
    lambda_vals = np.zeros((itermax + 1,))

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W, rank)

    scaled_lambda = lambda_vals[0] = (fscores[0] / gscores[0]) * _lambda

    best_score = np.Inf
    W_best = np.zeros((m, rank))
    H_best = np.zeros((rank, n))

    W_hat, H_hat = W, H
    for it in range(1, itermax + 1):
        # update H
        H_new = sep_update_H(M, W_hat, H)
        H_hat = non_neg(H_new + beta * (H_new - H))

        # update W
        W_new = sep_update_W(M, W, H_hat, scaled_lambda)
        W_hat = non_neg(W_new + beta * (W_new - W))

        fscores[it] = np.linalg.norm(M - W @ H, 'fro')
        gscores[it] = calculate_gscore(W, rank)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > fscores[it] + lambda_vals[it-1] * gscores[it]:
            W_hat, H_hat = W_new, H_new

            _beta = beta
            beta = beta / decay
        else:
            W_new, H_new = W_hat, H_hat

            _beta = min(1, _beta * _gr)
            beta = min(_beta, beta * gr)
        W, H = W_new, H_new

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores, gscores, np.r_[np.NaN, lambda_vals[1:]]


if __name__ == '__main__':
    np.random.seed(42)

    m, n, r_true = 12, 8, 3
    W_true = np.zeros((m, r_true))
    W_true[0:4, 0] = 1
    W_true[4:8, 1] = 1
    W_true[8:m, 2] = 1
    H_true = np.random.rand(r_true, n)
    H_true /= H_true.sum(axis=0, keepdims=True)
    M = W_true @ H_true

    r = 5
    W_ini = np.random.rand(m, r)
    H_ini = np.random.rand(r, n)
    Wb, Hb, Wl, Hl, fscores, gscores, lvals = nmf_son(M, W_ini.copy(), H_ini.copy(), _lambda=0.4, itermax=10,
                                                      andersen_type='z', andersen_win=3, verbose=True)
    print('done')
