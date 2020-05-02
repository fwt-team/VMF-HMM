# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: utils.py
@Time: 2020-03-02 16:39
@Desc: utils.py
"""
import numpy as np
import warnings
import scipy.io as scio
import matplotlib.pyplot as plt

from nilearn import datasets
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.special import iv
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, \
        adjusted_mutual_info_score as AMI, adjusted_rand_score as AR, silhouette_score as SI, calinski_harabasz_score as CH

from config import RESULT_DIR


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w


def vmf_pdf_log(x, mu, k):

    D = x.shape[len(x.shape) - 1]
    pdf = (D / 2 - 1) * np.log(k) - (D / 2) * np.log(2 * np.pi) - np.log(iv(D / 2 - 1, k)) + x.dot(mu.T * k)
    return pdf


def d_besseli(nu, kk):

    try:
        warnings.filterwarnings("ignore")
        bes = iv(nu + 1, kk) / (iv(nu, kk) + np.exp(-700)) + nu / kk
        assert (min(np.isfinite(bes)))
    except:
        bes = np.sqrt(1 + (nu**2) / (kk**2))

    return bes


def d_besseli_low(nu, kk):

    try:
        warnings.filterwarnings("ignore")
        bes = iv(nu + 1, kk) / (iv(nu, kk) + np.exp(-700)) + nu / kk
        assert (min(np.isfinite(bes)))
    except:
        bes = kk / (nu + 1 + np.sqrt(kk**2 + (nu + 1)**2)) + nu / kk

    return bes


def dirrnd(pu, n):

    samples = np.random.gamma(pu, 1)
    samples = samples / (np.sum(samples, 1, keepdims=True))

    return samples


def console_log(pred, data=None, labels=None, model_name='cluster', newJ=None, verbose=1):

    measure_dict = dict()
    if data is not None:
        measure_dict['si'] = SI(data, pred)
        measure_dict['ch'] = CH(data, pred)
    if labels is not None:
        measure_dict['acc'] = cluster_acc(pred, labels)[0]
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['ar'] = AR(labels, pred)
        measure_dict['ami'] = AMI(labels, pred)
    if newJ is not None:
        measure_dict['new_component'] = newJ

    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict


def get_syn_data(data_dir, data_name):

    datas = scio.loadmat('{}/{}.mat'.format(data_dir, data_name))
    data = datas['data']
    labels = datas['z'].reshape(-1)

    return data, labels


def get_adhd_data(data_dir='./datas/brain', n_subjects=1):

    dataset = datasets.fetch_adhd(data_dir=data_dir, n_subjects=n_subjects)
    imgs = dataset.func

    return imgs


def get_nyu_data(data_dir='./datas/brain', n_subjects=1):

    dataset = datasets.fetch_nyu_rest(n_subjects=n_subjects, data_dir=data_dir)
    imgs = dataset.func

    return imgs


# from nilearn.input_data import NiftiMasker
# from nilearn.masking import apply_mask
# mask = datasets.load_mni152_brain_mask()
# dataset = datasets.fetch_nyu_rest(data_dir='./datas/brain', n_subjects=1)
# data = apply_mask(dataset.func[0], mask_img=mask)
# print(1)


def get_haxby_data(data_dir='./datas/brain', subject_num=0, shuffle=True, data_split=100):

    # 0: 9 / 1: 6 / 2: 12 / 3: 5
    np.random.seed(9)
    data = scio.loadmat('{}/{}_haxby.mat'.format(data_dir, subject_num))
    labels = data['z'].reshape(-1)
    data = data['data']

    if shuffle:
        # shuffle data
        index = [i for i in range(data.shape[0])]
        np.random.shuffle(index)
        data = data[index]
        labels = labels[index]

    nor_test_data = data[:data_split]
    test_data = nor_test_data / np.linalg.norm(nor_test_data, axis=1, keepdims=True)
    test_data = test_data[np.newaxis, :, :]
    test_labels = labels[:data_split]

    nor_data = data[data_split:]
    train_data = nor_data / np.linalg.norm(nor_data, axis=1, keepdims=True)
    train_data = train_data[np.newaxis, :, :]

    # scio.savemat('{}/{}_haxby_pro.mat'.format(data_dir, subject_num), {'data': nor_data, 'test_data': nor_test_data, 'labels': labels, 'test_labels': test_labels})

    return train_data, test_data, nor_data, nor_test_data, labels, test_labels


def plot_data(each_data, time, data_name=None, save=False):
    """
    plot different cluster graph
    :param each_data:
    :param time:
    :param save:
    :return: None
    """
    lens = len(each_data)
    fig, axs = plt.subplots(nrows=int(lens / 2) if lens % 2 == 0 else int(lens / 2 + 1), ncols=2, constrained_layout=False)

    for index, ax in enumerate(axs.flat):
        if lens % 2 != 0 and index == lens:
            fig.delaxes(ax)
            break
        ax.plot(time, each_data[index].T)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Expression', fontsize=12)
        ax.set_title('Cluster: {}'.format(index + 1), fontsize=14)
    plt.show()

    # labels = np.concatenate([np.ones(data.shape[0]) * index for index, data in enumerate(each_data)]).astype(np.int)
    # tsne(X=np.vstack((i for i in each_data)), Y=labels)
    if save:
        fig.savefig('{}/cluster_result_{}.png'.format(RESULT_DIR, data_name))


def vmf_logp_xn_given_zn(data, _pi, xi, u, v):

    (T, D) = data.shape # data_num, data_dim
    (N, K, _) = xi.shape # state_num, mix_num
    logp_xn_given_zn = np.zeros((N, T))
    for i in range(N):
        logp_xn_given_zn[i] = LogVMFMMpdf(data, _pi[i], xi[i], u[i], v[i])

    return logp_xn_given_zn


def LogVMFMMpdf(x, _pi, mu, u, v):

    k = u / v
    D = x.shape[len(x.shape) - 1]
    pdf = np.log(_pi) + ((D / 2 - 1) * np.log(k) - (D / 2) * np.log(2 * np.pi) - np.log(iv(D / 2 - 1, k)) + x.dot(mu.T * k))

    logp_x = np.log(np.sum(np.exp(pdf - np.max(pdf, axis=1, keepdims=True)), 1)) + np.max(pdf, axis=1)
    return logp_x


def LogForwardBackward(logp_xn_given_zn, pi, A):
    """
    :param logp_xn_given_zn: N * T
    :param pi: N
    :param A: N * N
    :return: loggamma, logksi, loglik
    """
    (N, T) = logp_xn_given_zn.shape
    logalpha = np.zeros((N, T))
    logbeta = np.zeros((N, T))
    logc = np.zeros(T)
    # loggamma = np.zeros((N, T))
    # logksi = np.zeros((N, N, T))

    # init log alpha(z1), log beta(zN), c(1)
    tmp = logp_xn_given_zn[:, 0] + np.log(pi)
    logc[0] = np.log(np.sum(np.exp(tmp - np.max(tmp)))) + np.max(tmp)
    logalpha[:, 0] = -logc[0] + logp_xn_given_zn[:, 0] + np.log(pi)
    logbeta[:, T-1] = 0

    # calculate logalpha, c
    temp = np.log(A)[:, :, np.newaxis] + logalpha[:, :T - 1][:, np.newaxis, :] + \
           logp_xn_given_zn[:, 1:][np.newaxis, :, :]
    logc[1:] = np.log(np.sum(np.sum(np.exp(temp - np.max(temp.reshape((N * N, -1)), axis=0)), 0), 0)) + np.max(
        temp.reshape((N * N, -1)), axis=0)

    temp2 = logalpha[:, :T-1][:, np.newaxis, :] + np.log(A)[:, :, np.newaxis]
    # temp2[np.isinf(temp2)] = -np.inf
    logalpha[:, 1:] = -logc[1:] + logp_xn_given_zn[:, 1:] + np.log(
        np.sum(np.exp(temp2 - np.max(temp2, axis=0, keepdims=True)), 0)) + np.max(temp2, axis=0)

    # calculate logbeta
    temp3 = (logbeta[:, 1:] + logp_xn_given_zn[:, 1:])[np.newaxis, :, :] + np.log(A)[:, :, np.newaxis]
    logbeta[:, :T-1] = -logc[1:] + np.log(np.sum(np.exp(temp3 - np.max(temp3, axis=1, keepdims=True)), 1)) + np.max(
            temp3, axis=1)

    # calculate loggamma
    loggamma = logalpha + logbeta

    # calculate logksi
    logksi = np.log(A)[:, :, np.newaxis] + (-logc[1:] + logalpha[:, :T - 1])[:, np.newaxis, :] + (logp_xn_given_zn[:,
                                                                                                  1:] + logbeta[:, 1:])[
                                                                                                 np.newaxis, :, :]

    # calculate likelihood
    loglik = np.sum(logc)

    return loggamma, logksi, loglik


def Get_logp_xn_given_vn(x, mu, u, v):
    """
    :param x:
    :param C:
    :param mu:
    :param u:
    :param v:
    :return: logp_xn_given_vn
    """
    (T, D) = x.shape
    (N, K) = u.shape

    logp_xn_given_vn = np.zeros((N, K, T))
    kappa = u / v
    for i in range(N):
        logp_xn_given_vn[i] = vmf_pdf_log(x, mu[i], kappa[i]).T

    return logp_xn_given_vn


def CalculateLogIta(logp_xn_given_vn, pi, A, C):
    """
    :param logp_xn_given_vn: N * K * T
    :param pi: 
    :param A: 
    :param C: 
    :return: LogIta
    """""
    [N, K, T] = logp_xn_given_vn.shape
    logc = np.zeros(T)
    logalpha = np.zeros((N, K, T))
    logbeta = np.zeros((N, K, T))
    # logita = np.zeros((N, K, T))

    tmp = np.log(C) + logp_xn_given_vn[:, :, 0] + np.log(pi.T)
    logc[0] = np.log(np.sum(np.sum(np.exp(tmp - np.max(tmp)), 0), 0)) + np.max(tmp)
    logalpha[:, :, 0] = -logc[0] + tmp
    logbeta[:, :, T-1] = 0

    # calculate c, alpha
    for t in range(1, T):
        temp = np.zeros((N, K, N, K))
        for i in range(N):
            for k in range(K):
                temp[i, k] = logp_xn_given_vn[i, k, t] + np.log(C) + logalpha[:, :, t - 1] + np.log(
                    A[:, i][:, np.newaxis])
        logc[t] = np.log(np.sum(np.exp(temp - np.max(temp)))) + np.max(temp)

        for i in range(N):
            temp2 = logalpha[:, :, t - 1] + np.log(A[:, i][:, np.newaxis])
            if np.isinf(np.max(temp2)):
                logalpha[i, :, t] = -np.inf
            logalpha[i, :, t] = -logc[t] + logp_xn_given_vn[i, :, t] + np.log(C[i, :]) + np.log(
                np.sum(np.sum(np.exp(temp2 - np.max(temp2)), 0), 0)) + np.max(temp2)

    # calculate beta
    for t in reversed(range(T-1)):
        for i in range(N):
            temp3 = logbeta[:, :, t + 1] + logp_xn_given_vn[:, :, t + 1] + np.log(C) + np.log(A[i][:, np.newaxis])
            logbeta[i, :, t] = -logc[t + 1] + np.log(np.sum(np.sum(np.exp(temp3 - np.max(temp3)), 0), 0)) + np.max(temp3)

    return logalpha + logbeta


def UniformLogGammaKsi(LogGamma, LogKsi):
    """
    :param LogGamma: J * N * T
    :param LogKsi: J * N * N * T-1
    :return: Gamma, Ksi
    """
    J = LogGamma.shape[0]
    N = LogGamma[0].shape[0]

    for i in range(N):
        max_gamma_ary = np.zeros(J)
        max_ksi_ary = np.zeros(J)

        for j in range(J):
            max_gamma_ary[j] = np.max(LogGamma[j][i])
            max_ksi_ary[j] = np.max(LogKsi[j][:, i, :].reshape(-1))
        max_gamma = np.max(max_gamma_ary)
        max_ksi = np.max(max_ksi_ary)

        for j in range(J):
            LogGamma[j][i] = LogGamma[j][i] - max_gamma
            LogKsi[j][:, i, :] = LogKsi[j][:, i, :] - max_ksi

    return np.exp(LogGamma), np.exp(LogKsi)


def UniformLogIta(LogIta):
    """
    :param LogIta: J * N * K * T
    :return: ita
    """
    J = LogIta.shape[0]
    N = LogIta[0].shape[0]

    for i in range(N):
        max_ita_ary = np.zeros(J)

        for j in range(J):
            max_ita_ary[j] = np.max(np.max(LogIta[j][i], axis=1))
        max_ita = np.max(max_ita_ary)

        for j in range(J):
            LogIta[j][i] = LogIta[j][i] - max_ita

    return np.exp(LogIta)


def M_step_common(Gamma, Ksi):
    """
    :param Gamma: J * N * T
    :param Ksi: J * N * N * T-1
    :return:
    """
    J = Gamma.shape[0]
    N = Gamma[0].shape[0]

    # calculate p_start
    p_start_numer = np.zeros(N)
    for j in range(J):
        p_start_numer = p_start_numer + Gamma[j][:, 0]

    p_start = p_start_numer / np.sum(p_start_numer)

    # calculate A
    A_numer = np.zeros((N, N))
    for j in range(J):
        A_numer = A_numer + np.sum(Ksi[j], 2)
    A = A_numer / np.sum(A_numer, 1, keepdims=True)

    return p_start, A



