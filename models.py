# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: models.py
@Time: 2020-03-02 16:34
@Desc: models.py
"""
import numpy as np
import warnings

from scipy.special import digamma

from utils import d_besseli_low, d_besseli, dirrnd, vmf_logp_xn_given_zn, LogForwardBackward, \
    Get_logp_xn_given_vn, CalculateLogIta, UniformLogGammaKsi, UniformLogIta, M_step_common
from vmfmix.vmf import VMFMixture


class VmfHMM:

    def __init__(self, args):

        self.args = args
        self.N = args.N # the number of state
        self.K = args.K # the components of mixture
        self.T = 0 # the number of data
        self.D = 0 # data dimension
        self.J = 0
        self.log_lik = 0

        self.phi_pi = None # (N)
        self.phi_A = None # (N, N)
        self.phi_C = None # (N, K)

        self.o_pi = None # (N)
        self.o_A = None # (N, N)
        self.o_C = None # (N, N)
        self.ita = None

        self.w_pi = None # (N)
        self.w_A = None # (N, N)
        self.w_C = None # (N, K)

        # HMM params
        self.pi = None # (N)
        self.A = None # (N, N)
        self.C = None # (N, K)

        # VMF params
        self.zeta = None # (N, K)
        self.xi = None # (N, K, D)
        self.k = None # (N, K)
        self.u = None # (N, K)
        self.v = None # (N, K)
        self._pi = None # (N, K)
        self.prior = None

    def init_params(self, data):

        (self.T, self.D) = data[0].shape
        self.J = data.shape[0]

        # hyper params
        self.phi_pi = np.ones((1, self.N)) / self.N
        self.phi_A = np.ones((1, self.N)) / self.N
        self.phi_C = np.ones((1, self.K)) / self.K
        self.prior = {
            'mu': [np.sum(data[j], 0) / np.linalg.norm(np.sum(data[j], 0)) for j in range(self.J)],
            'zeta': 0.01,
            'u': 0.5,
            'v': 0.01,
        }

        # init model params
        self.o_pi = dirrnd(self.phi_pi, 1) * self.N * self.K
        self.o_A = np.ones((self.N, self.N))
        self.o_C = np.ones((self.N, self.K))
        for i in range(self.N):
            self.o_A[i] = dirrnd(self.phi_A, 1) * self.T
            self.o_C[i] = dirrnd(self.phi_C, 1) * self.T

        self.xi = np.ones((self.N, self.K, self.D))
        self.u = np.ones((self.N, self.K))
        self.v = np.ones((self.N, self.K))
        self._pi = np.ones((self.N, self.K)) / self.K

        warnings.filterwarnings('ignore')
        model = VMFMixture(n_cluster=(self.K * self.N), max_iter=300).fit(np.vstack((i for i in data))).get_params()
        self._pi = model['pi'].reshape((self.N, self.K))
        self.xi = model['xi'].reshape((self.N, self.K, -1))
        self.u = model['u'].reshape((self.N, self.K))
        self.v = model['v'].reshape((self.N, self.K))

    def check_data(self, data):

        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]

        return data

    def fit(self, data):

        data = self.check_data(data)
        self.init_params(data)
        self.vb_params(data)
        return self

    def fit_predict(self, data):

        data = self.check_data(data)
        self.fit(data)
        pred = self._predict(self.ita)
        return pred, self.ita

    def predict(self, data):

        data = self.check_data(data)
        _, _, _, ita, _ = self.forwback(data)
        pred = self._predict(ita)
        return pred, ita

    def _predict(self, ita):

        pred = []
        for j in range(self.J):
            temp = ita[j].reshape((self.N * self.K, -1)).T
            pred = np.concatenate((pred, np.argmax(temp, axis=1)))
        return pred.astype(np.int)

    def vb_params(self, data):

        pre_l = -np.inf
        converge = self.args.converge
        for ite in range(self.args.max_iter):

            # M-step
            self.update_hmm()

            # E-step
            self.o_pi, self.o_A, self.o_C, self.ita, log_l = self.forwback(data)
            # update vmf params
            self.update_vmf(data)

            # compute log_likelihood
            _converge = log_l - pre_l
            if self.args.verbose:
                print('iter: {}, log_likelihood: {}, converge: {}'.format(ite, log_l, _converge))
            if _converge < np.log(converge):
                if self.args.verbose:
                    print("xi: {} \nkappa: {} \n o_pi: {} \n o_A: {} \n o_C: {} \n".format(self.xi, self.u / self.v,
                                                                                           self.o_pi, self.o_A,
                                                                                           self.o_C))
                break
            pre_l = log_l

            if ite == self.args.max_iter - 1:
                print("xi: {} \nkappa: {} \n o_pi: {} \n o_A: {} \n o_C: {} \n".format(self.xi, self.u / self.v,
                                                                                   self.o_pi, self.o_A,
                                                                                   self.o_C))

    def update_vmf(self, data):

        D = self.D
        self.k = self.u / self.v

        temp = 0
        temp1 = 0
        for j in range(self.J):
            temp = temp + self.prior['zeta'] * self.prior['mu'][j] + self.ita[j].dot(data[j])
            temp1 = temp1 + self.ita[j]
        self.zeta = np.linalg.norm(temp, axis=2, keepdims=True)
        self.xi = temp / self.zeta
        self.u = self.prior['u'] + (D / 2 - 1) * np.sum(temp1, 2) + self.zeta.reshape((self.N, self.K)) * self.k * (
                           d_besseli_low(D / 2 - 1, self.zeta.reshape((self.N, self.K)) * self.k))
        self.v = self.prior['v'] + d_besseli(D / 2 - 1, self.k) * np.sum(temp1, 2) + \
                       self.prior['zeta'] * (
                           d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))

    def update_hmm(self):

        self.w_pi = self.phi_pi + self.o_pi
        self.w_A = self.phi_A + self.o_A
        self.w_C = self.phi_C + self.o_C

        self.pi = np.exp(digamma(self.w_pi) - digamma(np.sum(self.w_pi, keepdims=True)))
        self.A = np.exp(digamma(self.w_A) - digamma(np.sum(self.w_A, 1, keepdims=True)))
        self.C = np.exp(digamma(self.w_C) - digamma(np.sum(self.w_C, 1, keepdims=True)))

    def forwback(self, data):

        (J, T, D) = data.shape
        LogGamma = np.ones((J, self.N, T))
        LogKsi = np.ones((J, self.N, self.N, T - 1))
        Loglik = np.ones(J)
        LogIta = np.ones((J, self.N, self.K, T))
        for j in range(self.J):
            logp_xn_given_zn = vmf_logp_xn_given_zn(data[j], self._pi, self.xi, self.u, self.v)
            (LogGamma[j], LogKsi[j], Loglik[j]) = LogForwardBackward(logp_xn_given_zn, self.pi, self.A)
            logp_xn_given_vn = Get_logp_xn_given_vn(data[j], self.xi, self.u, self.v)
            LogIta[j] = CalculateLogIta(logp_xn_given_vn, self.pi, self.A, self.C)

        [Gamma, Ksi] = UniformLogGammaKsi(LogGamma, LogKsi)
        ita = UniformLogIta(LogIta)
        [p_start, A] = M_step_common(Gamma, Ksi)

        C_nomer = np.zeros((self.N, self.K))
        C_denom = np.zeros((self.N, 1))
        for j in range(self.J):
            C_nomer = C_nomer + np.sum(ita[j], 2)
            C_denom = C_denom + np.sum(np.sum(ita[j], 1), 1, keepdims=True)

        C = C_nomer / C_denom

        return p_start, A, C, ita, np.sum(Loglik)


