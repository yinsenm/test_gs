import pandas as pd
import numpy as np
from numpy import linalg as LA


def is_psd_def(cov_mat):
    """
    :param cov_mat: covariance matrix of p x p
    :return: true if positive semi definite (PSD)
    """
    return np.all(np.linalg.eigvals(cov_mat) > -1e-6)


def correlation_from_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """
    :param covariance: covariance matrix as input
    :return: correlation matrix
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def ledoit(df_rets: pd.DataFrame) -> pd.DataFrame:
    """
    compute Ledoit covariance Statistics
    :param df_rets: assets return matrix of dimension n x p
    :return: Ledoit covariance matrix of p x p
    """
    symbols = df_rets.columns
    x = df_rets.values.copy()
    t, n = x.shape
    _mean = np.tile(x.mean(axis=0), (t, 1))

    # de-mean the returns
    x -= _mean

    # compute sample covariance matrix
    sample = (1 / t) * x.transpose() @ x

    # compute the prior
    _var = np.diag(sample)
    sqrt_var = np.sqrt(_var).reshape((-1, n))
    rBar = (np.sum(sample / (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))) - n) / (n * (n - 1))
    prior = rBar * (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))
    prior[np.diag_indices_from(prior)] = _var.tolist()

    # compute shrinkage parameters and constant
    # what we call pi-hat
    y = x ** 2
    phiMat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
    phi = np.sum(phiMat)

    # what we call rho-hat
    term1 = (x ** 3).T @ x / t
    help = (x.T @ x) / t
    helpDiag = np.diag(help).reshape((n, 1))
    term2 = np.tile(helpDiag, (1, n)) * sample
    term3 = help * np.tile(_var.reshape(n, 1), (1, n))
    term4 = np.tile(_var.reshape(n, 1), (1, n)) * sample
    thetaMat = term1 - term2 - term3 + term4
    thetaMat[np.diag_indices_from(thetaMat)] = np.zeros(n)
    rho = np.sum(np.diag(phiMat)) + rBar * np.sum(((1 / sqrt_var.T).dot(sqrt_var)) * thetaMat)

    # what we call gamma-hat
    gamma = LA.norm(sample - prior, 'fro') ** 2

    # compute shrinkage costant
    kappa = (phi - rho) / gamma
    shrinkage = max(0, min(1, kappa / t))

    # compute the estimator
    covMat = shrinkage * prior + (1 - shrinkage) * sample
    return pd.DataFrame(covMat, index=symbols, columns=symbols)

def gerber_cov_stat1(df_rets: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    compute Gerber covariance Statistics 1
    :param df_rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    rets = df_rets.values
    symbols = df_rets.columns
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return pd.DataFrame(cov_mat, index=symbols, columns=symbols)


def gerber_cov_stat2(df_rets: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    compute Gerber covariance Statistics 2
    :param df_rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: dataframe of Gerber covariance or correlation matrix of dimension p x p
    """
    symbols = df_rets.columns
    rets = df_rets.values
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    U = np.copy(rets)
    D = np.copy(rets)

    # update U and D matrix
    for i in range(p):
        U[:, i] = U[:, i] >= sd_vec[i] * threshold
        D[:, i] = D[:, i] <= -sd_vec[i] * threshold

    # update concordant matrix
    N_CONC = U.transpose() @ U + D.transpose() @ D

    # update discordant matrix
    N_DISC = U.transpose() @ D + D.transpose() @ U
    H = N_CONC - N_DISC
    h = np.sqrt(H.diagonal())

    # reshape vector h and sd_vec into matrix
    h = h.reshape((p, 1))
    sd_vec = sd_vec.reshape((p, 1))

    cor_mat = H / (h @ h.transpose())
    cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
    return pd.DataFrame(cov_mat, index=symbols, columns=symbols)