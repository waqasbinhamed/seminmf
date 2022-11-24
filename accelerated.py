import numpy as np

TOL = 1e-12


def non_neg(arr):
    arr[arr < 0] = 0
    return arr


def calculate_gscore(W, r):
    gscore = 0
    for i in range(r):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def AA(R_history):
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


def update_wj_z_acc(W, Mj, z, hj, j, m, r, _lambda, aa_size=-1, itermax=1000):
    def calculat_z(Mj, _lambda, ci_arr, hj, hj_norm_sq, new_wi_arr, num_edges, rho, y0, yf, yi_arr, z):
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

        return new_w0, new_wf, new_z

    rho = 1
    num_edges = (r * (r - 1)) / 2
    hj_norm_sq = np.linalg.norm(hj) ** 2
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    yi_arr = np.random.rand(m, r - 1)
    yf = np.random.rand(m, 1)
    y0 = np.random.rand(m, 1)

    new_w0, new_wf, new_z = calculat_z(Mj, _lambda, ci_arr, hj, hj_norm_sq, new_wi_arr, num_edges, rho, y0, yf, yi_arr,
                                       z)

    new_yf = yf + rho * (new_wf - new_z)
    new_y0 = y0 + rho * (new_w0 - new_z)
    new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

    if aa_size != -1:
        Qz = new_z.copy()
        Rz = new_z - z

    it = 1
    while it < itermax and np.linalg.norm(new_z - z) / np.linalg.norm(z) > TOL:

        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_w0, new_wf, new_z = calculat_z(Mj, _lambda, ci_arr, hj, hj_norm_sq, new_wi_arr, num_edges, rho, y0, yf,
                                           yi_arr, z)

        if aa_size != -1:
            Rz = np.c_[Rz, new_z - z]
            Qz = np.c_[Qz, new_z]
            if Rz.shape[1] > aa_size:
                Rz = Rz[:, 1:]
                Qz = Qz[:, 1:]

            cz = AA(Rz)
            new_z = (Qz @ cz).reshape(m, 1)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
        it += 1
    return new_z


def update_wj_all_acc(W, Mj, z, hj, j, m, r, _lambda, aa_size=-1, itermax=1000):
    rho = 1
    num_edges = (r * (r - 1)) / 2
    hj_norm_sq = np.linalg.norm(hj) ** 2
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    yi_arr = np.random.rand(m, r - 1)
    yf = np.random.rand(m, 1)
    y0 = np.random.rand(m, 1)

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

    new_yf = yf + rho * (new_wf - new_z)
    new_y0 = y0 + rho * (new_w0 - new_z)
    new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

    if aa_size != -1:
        Qz = new_z.copy()
        Rz = new_z - z

        Qwf = new_wf.copy()
        Rwf = new_wf - z

        Qw0 = new_w0.copy()
        Rw0 = new_w0 - z

        Qwi_arr = new_wi_arr.copy()
        Rwi_arr = new_wi_arr - z

    it = 1
    while it < itermax and np.linalg.norm(new_z - z) / np.linalg.norm(z) > TOL:

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

        if aa_size != -1:
            Rwf = np.c_[Rwf, new_wf - z]
            Qwf = np.c_[Qwf, new_wf]
            if Rwf.shape[1] > aa_size:
                Rwf = Rwf[:, 1:]
                Qwf = Qwf[:, 1:]

            cwf = AA(Rwf)
            new_wf = (Qwf @ cwf).reshape(m, 1)

            Rw0 = np.c_[Rw0, new_w0 - z]
            Qw0 = np.c_[Qw0, new_w0]
            if Rw0.shape[1] > aa_size:
                Rw0 = Rw0[:, 1:]
                Qw0 = Qw0[:, 1:]

            cw0 = AA(Rw0)
            new_w0 = (Qw0 @ cw0).reshape(m, 1)

            Rwi_arr = np.dstack((Rwi_arr, new_wi_arr - z))
            Qwi_arr = np.dstack((Qwi_arr, new_wi_arr))
            if Rwi_arr.shape[2] > aa_size:
                Rwi_arr = Rwi_arr[:, :, 1:]
                Qwi_arr = Qwi_arr[:, :, 1:]

            for i in range(r-1):
                cwi = AA(Rwi_arr[:, i, :])
                new_wi_arr[:, i: i+1] = (Qwi_arr[:, i, :] @ cwi).reshape(m, 1)

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if aa_size != -1:
            Rz = np.c_[Rz, new_z - z]
            Qz = np.c_[Qz, new_z]
            if Rz.shape[1] > aa_size:
                Rz = Rz[:, 1:]
                Qz = Qz[:, 1:]

            cz = AA(Rz)
            new_z = (Qz @ cz).reshape(m, 1)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
        it += 1
    return new_z


def nmf_son_aa(M, W, H, _lambda=0.0, itermax=1000, aa_type='z', aa_size=-1, scale_lambda=False, verbose=False):
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
            if aa_type == 'z':
                W[:, j: j + 1] = wj = update_wj_z_acc(W, Mj, wj, hj, j, m, r, scaled_lambda, aa_size=aa_size)
            elif aa_type == 'all':
                W[:, j: j + 1] = wj = update_wj_all_acc(W, Mj, wj, hj, j, m, r, scaled_lambda, aa_size=aa_size)

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


if __name__ == '__main__':
    np.random.seed(42)


    def zeros_mask(arr):
        m, n = arr.shape
        indices = np.random.choice(m * n, replace=False, size=int(m * n * 0.4))
        arr[np.unravel_index(indices, (m, n))] = 0
        return arr


    m, n, r_true = 12, 8, 3
    W_true = np.zeros((m, r_true))
    W_true[0:4, 0] = 1
    W_true[4:8, 1] = 1
    W_true[8:m, 2] = 1
    H_true = zeros_mask(np.random.rand(r_true, n))
    M = W_true @ H_true

    r = 5
    W_ini = np.random.rand(m, r)
    H_ini = np.random.rand(r, n)
    Wb, Hb, Wl, Hl, fscores, gscores, lvals = nmf_son_aa(M, W_ini, H_ini, _lambda=0.4, itermax=100, scale_lambda=True,
                                                         aa_type='z', aa_size=1, verbose=True)
    print('done')
