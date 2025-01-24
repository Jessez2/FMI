from typing import List
import numpy as np
import torch
from numpy import ndarray
from torch.optim import Adam
import warnings
from Utils.lbfgsb_scipy import LBFGSBScipy
from Model.FMargModel import MargLR_model_new_with_grid_search

warnings.filterwarnings("ignore")

def local_score_fmarg(Data: ndarray, Xi: int, PAi: List[int], param=None):
    var_idx = param['var_idx']
    X = torch.tensor(Data[:, Xi])
    if len(PAi):
        pa_ids_list = []
        data_idx = []
        idx = 0
        for i in PAi:
            PA_temp = var_idx[i]
            data_idx += PA_temp
            # current idx
            PA_temp_dim = len(PA_temp)
            temp_list = []
            for tt in range(idx, idx + PA_temp_dim):
                temp_list.append(tt)
            idx += PA_temp_dim
            pa_ids_list.append(temp_list)

        param['pa_list'] = pa_ids_list
        PA = torch.tensor(Data[:, PAi])
    else:
        PA = torch.tensor(np.zeros((Data.shape[0], 1)))
        param['pa_list'] = [[0]]

    # print("marg_lr_model")
    local_score_model = MargLR_model_new_with_grid_search(PA, X, param).to(param['device'])

    # Adam
    if param['optim'] == 'adam':
        param['epochs'] = 100
        # print(local_score_model.noise_scale_)
        optim = Adam([local_score_model.noise_scale_], lr=0.005)
        for i in range(param['epochs']):
            d_train = local_score_model.train_step(PA, X, optim)

    else:
        param['optim']=='grid_search':
        local_score_model.grid_search(PA,X,num_noises=500)
        
# if you choose grid search, please turn off the self.grad
    return local_score_model.score