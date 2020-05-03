# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn._utils.compat import Memory
from nilearn.plotting import plot_prob_atlas, show, plot_stat_map
from nilearn.image import iter_img
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

from utils import console_log
from config import RESULT_DIR


class ClusterProcess(MultiPCA):

    def __init__(self, model, mask=None, n_cluster=40, n_components=20, group=True, sub_num=1, smoothing_fwhm=6,
                 do_cca=True,
                 threshold='auto',
                 n_init=10,
                 random_state=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super(ClusterProcess, self).__init__(
            n_components=n_components,
            do_cca=do_cca,
            random_state=random_state,
            # feature_compression=feature_compression,
            mask=mask, smoothing_fwhm=smoothing_fwhm,
            standardize=standardize, detrend=detrend,
            low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args,
            memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

        self.n_cluster = n_cluster
        self.model_ = model
        self.group = group
        self.sub_num = sub_num
        self.train_data = None
        self.model = None

    def hvmf_fit(self, data):

        np.random.seed(0)
        self.train_data = data / np.linalg.norm(data, axis=1, keepdims=True)
        # self.train_data = (data - np.min(data, axis=1, keepdims=True)) / \
        #                   (np.max(data, axis=1) - np.min(data, axis=1))[:, np.newaxis]

        data = self.train_data[np.random.randint(0, data.shape[0], 10000), :]

        self.model = self.model_.fit(data)

        return self

    def _raw_fit(self, data):

        group = self.group
        sub_num = self.sub_num

        if group:
            data = MultiPCA._raw_fit(self, data).T
        else:
            data = data.reshape((3, self.n_components, -1))
            data = data[sub_num - 1]

        # plt.plot(data[:, :3])
        # plt.tight_layout()
        # plt.show()
        self.hvmf_fit(data.T)
        return self

    def plot_pro(self, ita, save=False, item_file='group', name='vmf', choose=None, cut_coords=None):

        for component in ita:
            if component.max() < -component.min():
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(ita)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            display.savefig('{}/brain/{}/{}/pro.png'.format(RESULT_DIR, name, item_file), dpi=200)

        for i, cur_img in enumerate(iter_img(components_img)):
            if cut_coords is not None:
                display = plot_stat_map(cur_img, cut_coords=cut_coords[i], dim=-.5, threshold=1e-3,
                                        cmap=plt.get_cmap('autumn'))
            else:
                display = plot_stat_map(cur_img, dim=-.5, threshold=1e-3,
                                        cmap=plt.get_cmap('autumn'))
            if save:
                if choose is not None:
                    display.savefig('{}/brain/{}/{}/item{}.png'.format(RESULT_DIR, name, item_file, choose[i] + 1), dpi=200)
                else:
                    display.savefig('{}/brain/{}/{}/item{}.png'.format(RESULT_DIR, name, item_file, i + 1), dpi=200)
        if save is False:
            show()

    def plot_all(self, pred, save=False, item_file='group', name='vmf'):

        data = np.zeros((self.n_cluster, pred.shape[0]))
        total = 0
        for i in range(self.n_cluster):
            data[i][pred != i] = 0
            data[i][pred == i] = 1
            total += data[i][data[i] != 0].shape[0]

        print(total)

        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(data)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            display.savefig('{}/brain/{}/{}/all.png'.format(RESULT_DIR, name, item_file), dpi=200)
        else:
            show()
