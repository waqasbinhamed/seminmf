import numpy as np
from nmf_son.utils import non_neg, calculate_gscore


TOL = 1e-4
INNER_TOL = 1e-6


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

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
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

            for i in range(r - 1):
                cwi = find_andersen_coeff(wi_residuals_arr[:, i, :])
                new_wi_arr[:, i: i + 1] = (prev_wis_arr[:, i, :] @ cwi).reshape(m, 1)

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
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

        if andersen_win is not None:
            if it > 1:
                prev_yfs = np.c_[prev_yfs, new_yf]
                yf_residuals = np.c_[yf_residuals, new_yf - yf]
                if yf_residuals.shape[1] > andersen_win:
                    prev_yfs = prev_yfs[:, 1:]
                    yf_residuals = yf_residuals[:, 1:]

                cyf = find_andersen_coeff(yf_residuals)
                new_yf = (prev_yfs @ cyf).reshape(m, 1)

                prev_y0s = np.c_[prev_y0s, new_y0]
                y0_residuals = np.c_[y0_residuals, new_y0 - y0]
                if y0_residuals.shape[1] > andersen_win:
                    prev_y0s = prev_y0s[:, 1:]
                    y0_residuals = y0_residuals[:, 1:]

                cy0 = find_andersen_coeff(y0_residuals)
                new_y0 = (prev_y0s @ cy0).reshape(m, 1)

                prev_yis_arr = np.dstack((prev_yis_arr, new_yi_arr))
                yi_residuals_arr = np.dstack((yi_residuals_arr, new_yi_arr - yi_arr))
                if yi_residuals_arr.shape[2] > andersen_win:
                    prev_yis_arr = prev_yis_arr[:, :, 1:]
                    yi_residuals_arr = yi_residuals_arr[:, :, 1:]

                for i in range(r - 1):
                    cyi = find_andersen_coeff(yi_residuals_arr[:, i, :])
                    new_yi_arr[:, i: i + 1] = (prev_yis_arr[:, i, :] @ cyi).reshape(m, 1)

            else:
                prev_yfs = new_yf.copy()
                yf_residuals = new_yf - yf

                prev_y0s = new_y0.copy()
                y0_residuals = new_y0 - y0

                prev_yis_arr = new_yi_arr.copy()
                yi_residuals_arr = new_yi_arr - yi_arr
    return new_z


def nmf_son_z_accelerated(M, W, H, _lambda=0.0, itermax=1000, andersen_win=2, early_stop=False, verbose=False):
    """Calculates NMF decomposition of the M matrix with andersen acceleration options."""
    m, n = M.shape
    r = W.shape[1]

    fscores = np.full((itermax + 1,), np.NaN)
    gscores = np.full((itermax + 1,), np.NaN)
    lambda_vals = np.full((itermax + 1,), np.NaN)

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W)

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
            W[:, j: j + 1] = wj = update_wj_andersen_z(W, Mj, wj, hj, j, scaled_lambda, andersen_win=andersen_win)
            Mj = Mj - wj @ hj

        fscores[it] = np.linalg.norm(M - W @ H, 'fro')
        gscores[it] = calculate_gscore(W)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        if early_stop:
            old_score = fscores[it - 1] + lambda_vals[it - 2] * gscores[it - 1]
            print(abs(old_score - total_score) / old_score)
            if abs(old_score - total_score) / old_score < TOL:
                break

        scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores, gscores, np.r_[np.NaN, lambda_vals[1:]]


def nmf_son_all_accelerated(M, W, H, _lambda=0.0, itermax=1000, andersen_win=2, early_stop=False, verbose=False):
    """Calculates NMF decomposition of the M matrix with andersen acceleration options."""
    m, n = M.shape
    r = W.shape[1]

    fscores = np.full((itermax + 1,), np.NaN)
    gscores = np.full((itermax + 1,), np.NaN)
    lambda_vals = np.full((itermax + 1,), np.NaN)

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W)

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
            W[:, j: j + 1] = wj = update_wj_andersen_all(W, Mj, wj, hj, j, scaled_lambda, andersen_win=andersen_win)
            Mj = Mj - wj @ hj

        fscores[it] = np.linalg.norm(M - W @ H, 'fro')
        gscores[it] = calculate_gscore(W)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        if early_stop:
            old_score = fscores[it - 1] + lambda_vals[it - 2] * gscores[it - 1]
            print(abs(old_score - total_score) / old_score)
            if abs(old_score - total_score) / old_score < TOL:
                break

        scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores, gscores, np.r_[np.NaN, lambda_vals[1:]]
