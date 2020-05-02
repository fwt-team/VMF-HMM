# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train.py
@Time: 2020-03-02 16:28
@Desc: train.py
"""
try:
    import argparse
    import numpy as np
    import torch
    import pandas as pd

    from scipy import io as scio
    from nilearn import datasets
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from sklearn.cluster import KMeans

    from models import VmfHMM
    from config import DATA_PARAMS, DATASETS_DIR, SYN_DIR, BRAIN_DIR
    from utils import console_log, plot_data, get_adhd_data, get_haxby_data, get_syn_data, get_nyu_data
    from cluster_process import ClusterProcess
    from vmfmix.vmf import VMFMixture

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        self.model = VmfHMM(args)

    def train(self, data):

        return self.model.fit_predict(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='VMF-HMM', description='')

    parser.add_argument('-data_name', '--data_name', dest='data_name', help='data_name', default='adhd')
    parser.add_argument('-c', '--data_choose', dest='data_choose', help='data_choose', default=2)
    parser.add_argument('-N', '--state', dest='N', help='the number of state', default=2)
    parser.add_argument('-K', '--component', dest='K', help='the number of component', default=3)
    parser.add_argument('-max_iter', '--max_iter', dest='max_iter', help='max_iter', default=100)
    parser.add_argument('-converge', '--converge', dest='converge', help='converge', default=1+1e-4)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    args = parser.parse_args()

    # ====================test===================== #
    if args.data_choose == 0:
        args.N = 2
        args.K = 2
        data, labels = get_syn_data(SYN_DIR, 'small_data')
        trainer = Trainer(args)
        pred, _ = trainer.train(data[np.newaxis, :, :])
        console_log(pred=pred, labels=labels, data=np.vstack((i for i in data)), model_name='HMM-VMF-test')

    # ====================brain===================== #
    if args.data_choose == 2:
        # ========================================================================================== #
        # group: n_s=30, n_c=40,  sf=6
        # subj:  n_s=3,  n_c=176, sf=10
        args.N = 5
        args.K = 8
        func_filenames = get_adhd_data(data_dir=BRAIN_DIR, n_subjects=30)
        cp = ClusterProcess(model=VmfHMM(args), n_cluster=args.N*args.K, n_components=40, group=True,
                            sub_num=3, smoothing_fwhm=6., memory="nilearn_cache", threshold=1., memory_level=2,
                            verbose=10, random_state=0)
        cp.fit(func_filenames)

        # pred = cp.model._predict(cp.model.ita)
        # pred, ita = cp.model.predict(cp.train_data)
        # ita = ita.reshape((args.N * args.K, -1))

        # ita = np.exp(cp.model._estimate_weighted_log_prob(cp.train_data))
        # ita = (ita / np.sum(ita, axis=1, keepdims=True)).T
        # pred = np.argmax(ita.T, axis=1)

        # pred = cp.model.predict(cp.train_data)

        # scio.savemat('./datas/brain/adhd/pred/test_hmm_vmf_sb1.mat', {'pred': pred, 'result': ita})
        # vmfs = np.array([23,33,40,20,17,36,12]) - 1
        # gmms = np.array([3,4,8,9,18,28,38]) - 1
        # cut_coords = np.array([
        #     [57, -5, 28],
        #     [1, -84, 28],
        #     [1, -60, -48],
        #     [-36, 0, -30],
        #     [0, -38, -52],
        #     [-1, 32, -12],
        #     [1, -53, 28],
        # ])
        pre_data = scio.loadmat('./datas/brain/adhd/pred/test_hmm_gmm.mat')
        pred = pre_data['pred'].reshape(-1)
        # ita = pre_data['result'][vmfs]
        # ita[ita > 0.25] = 0
        #
        cp.plot_pro(ita, save=False, item_file='var', name='vmf', choose=vmfs, cut_coords=cut_coords)
        cp.plot_all(pred, save=False, item_file='sub1', name='vmf')

        ca = np.unique(pred)
        print(ca)
        console_log(pred=pred[:5000], data=cp.train_data[:5000], model_name='HMM-VMF-brain')
