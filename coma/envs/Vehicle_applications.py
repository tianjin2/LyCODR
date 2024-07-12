
import scipy.stats as stats
# 导入 numpy 模块，这是 Python 的主要科学计算库
import numpy as np
# 从 sac 环境的常量模块中导入所有常量
from coma.envs import constant as ct


veh_app_info = {

    ct.SPEECH_RECOGNITION: {
        'workload': 10435,
        'popularity': 0.5,
        'min_bits': 40 * ct.KB,
        'max_bits': 300 * ct.KB,
    },

    ct.NLP: {
        'workload': 25346,
        'popularity': 0.8,
        'min_bits': 4 * ct.KB,
        'max_bits': 100 * ct.KB,
    },

    ct.FACE_RECOGNITION: {
        'workload': 45043,
        'popularity': 0.4,
        'min_bits': 10 * ct.KB,
        'max_bits': 100 * ct.KB,
    },

    ct.SEARCH_REQ: {
        'workload': 8405,
        'popularity': 0.4,
        'min_bits': 2 * ct.BYTE,
        'max_bits': 100 * ct.BYTE,
    },

    ct.LANGUAGE_TRANSLATION: {
        'workload': 34252,
        'popularity': 1,
        'min_bits': 2 * ct.BYTE,
        'max_bits': 5000 * ct.BYTE,
    },

    ct.PROC_3D_GAME: {
        'workload': 54633,
        'popularity': 0.1,
        'min_bits': 0.1 * ct.MB,
        'max_bits': 3 * ct.MB,
    },

    ct.VR: {
        'workload': 40305,
        'popularity': 0.1,
        'min_bits': 0.1 * ct.MB,
        'max_bits': 3 * ct.MB,
    },

    ct.AR: {
        'workload': 34532,
        'popularity': 0.1,
        'min_bits': 0.1 * ct.MB,
        'max_bits': 3 * ct.MB,
    }
}


def veh_app_list():
    return veh_app_info.keys()


def veh_app_type_pop():

    return [(i, veh_app_info[i]['popularity']) for i in list(veh_app_info.keys())]


def get_info(type, info_name='workload'):
    return veh_app_info[type][info_name]


def arrival_bits(types, dist='deterministic', size=1):

    min_bits = veh_app_info[types]['min_bits']
    max_bits = veh_app_info[types]['max_bits']

    mu = (min_bits + max_bits) / 2
    sigma = (max_bits - min_bits) / 4


    if dist == 'normal':

        if size == 1:
            return int(stats.truncnorm.rvs((min_bits - mu) / sigma, (max_bits - mu) / sigma, loc=mu, scale=sigma))

        return stats.truncnorm.rvs((min_bits - mu) / sigma, (max_bits - mu) / sigma, loc=mu, scale=sigma,
                                   size=size).astype(int)
    elif dist == 'deterministic':

        return mu
    else:
        # 否则返回1
        return 1


def normal_dist(list, mu, sig, peak):
    return peak * np.exp(-((list - mu) ** 2) / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))



