# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2020-03-02 16:35
@Desc: config.py
"""

import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datas')

SYN_DIR = os.path.join(DATASETS_DIR, 'synthetic')

BRAIN_DIR = os.path.join(DATASETS_DIR, 'brain')

RESULT_DIR = os.path.join(REPO_DIR, 'results')

# difference datasets config
DATA_PARAMS = {}