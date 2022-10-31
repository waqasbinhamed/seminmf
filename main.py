import numpy as np


def initialize_matrices(m, n, r):
    return np.random.rand(m, r), np.random.rand(r, n)


def nmf(M, W, H, _lambda=0, iters=100):
    m, n = M.shape
    r = W.shape[1]

    Mj = M - W @ H
    for it in range(iters):
        for j in range(r):
            wj = W[:, j: j+1]
            hj = H[j: j+1, :]

            Mj = Mj + wj @ hj

            # update h_j
            tmp = wj.T @ Mj
            tmp[tmp < 0] = 0
            new_hj = tmp / (np.linalg.norm(wj) ** 2)
            H[j: j+1, :] = new_hj

            # TODO: update w_j
            new_wj = wj

            Mj = Mj - new_wj @ new_hj


if __name__ == '__main__':
    m, n, r = 20, 10, 5
    M0 = np.random.rand(m, n)
    W0, H0 = initialize_matrices(m, n, r)
    nmf(M0, W0, H0)
