import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Union, Optional
from scipy.linalg import sqrtm, inv
from scipy.spatial.distance import cdist
from numpy.random import default_rng

def pdinv_Array(A):
    numData = A.shape[0]
    try:
        U = torch.linalg.cholesky(A)
        invU = torch.eye(numData, device=A.device).mm(torch.linalg.inv(U))
        Ainv = invU.mm(invU.T)
    except RuntimeError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using SVD')
        u, s, vh = torch.linalg.svd(A)
        Ainv = vh.T.mm(torch.diag(1 / s)).mm(u.T)
    except Exception as e:
        raise e
    return Ainv


class GaussianKernel:
    """
    A class to find gaussian kernel evaluations k(x, y) = exp (-||x - y||^2/2 sigma^2)
    """

    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        j1 = np.ones((X1.shape[0], 1))
        j2 = np.ones((X2.shape[0], 1))

        diagK1 = np.sum(X1 ** 2, 1)
        diagK2 = np.sum(X2 ** 2, 1)

        X1X2 = np.dot(X1, X2.T)

        Q = (2 * X1X2 - np.outer(diagK1, j2) - np.outer(j1, diagK2)) / (2 * self.sigma ** 2)

        return np.exp(Q)

    def __str__(self):
        return "GaussianKernel: sigma = " + str(self.sigma)


class DiagGaussianKernel:
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        # if X1.shape[1] != X2.shape[1]:
        # raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        K = np.exp(- np.sum((X1 - X2) ** 2, 1) / (2 * self.sigma ** 2))
        K = np.array(K, ndmin=2).T
        return K


def kernel_icd(X, kernel, m=100, precision=1e-6):
    """Approximates a kernel matrix using incomplete Cholesky decomposition (ICD).

    Input:	- X: data matrix in row format (each data point is a row)
                - kernel: the kernel function. It should calculate on the diagonal!
                - kpar: vector containing the kernel parameters.
                - m: maximal rank of solution
                - precision: accuracy parameter of the ICD method
    Output:	- G: "narrow tall" matrix of the decomposition K ~= GG'
                - subset: indices of data selected for low-rank approximation

    USAGE: G = km_kernel_icd(X,ktype,kpar,m,precision)

    Based on code from Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2010.

    The algorithm in this file is based on the following publication:
    Francis R. Bach, Michael I. Jordan. "Kernel Independent Component
    Analysis", Journal of Machine Learning Research, 3, 1-48, 2002.

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, version 3 (as included and available at
    http://www.gnu.org/licenses).
    """
    n = X.shape[0]

    perm = np.arange(n)  # permutation vector
    d = np.zeros((1, n))  # diagonal of the residual kernel matrix
    G = np.zeros((n, m))
    subset = np.zeros((1, m))

    for i in range(m):
        x = X[perm[i: n + 1], :]
        if i == 0:  # diagonal of kernel matrix
            d[:, i: n + 1] = kernel(x, x).T
        else:  # update the diagonal of the residual kernel matrix
            d[:, i: n + 1] = kernel(x, x).squeeze() - np.sum(G[i: n + 1, : i] ** 2, 1).T

        dtrace = np.sum(d[:, i:n + 1])

        # if dtrace <= 0:
        #     print("Warning: negative diagonal entry: sdasdasda")

        if dtrace <= precision:
            G = G[:, :i]
            subset = subset[:i]
            break

        m2 = np.max(d[:, i: n + 1])  # find the new best element
        j = np.argmax(d[:, i: n + 1])
        # j = j + i - 1  #take into account the offset i
        j = j + i  # take into account the offset i
        m1 = np.sqrt(m2)
        subset[0, i] = j

        perm[[i, j]] = perm[[j, i]]  # permute elements i and j
        # permute rows i and j
        G[[i, j], :i] = G[[j, i], :i]
        G[i, i] = m1  # new diagonal element

        # Calculate the i-th column. May
        # introduce a slight numerical error compared to explicit calculation.

        G[i + 1: n + 1, i] = (kernel(X[perm[i], :], X[perm[i + 1:n + 1], :]).T -
                              np.dot(G[i + 1:n + 1, :i], G[i, :i].T)) / m1

    ind = np.argsort(perm)
    G = G[ind, :]
    return G


def feat_gen(X, wx):
    """
    Generates features by dividing the data X by the vector wx.

    :param X: Input data matrix (torch tensor).
    :param wx: Feature scaling vector (torch tensor).
    :return: Scaled feature matrix (torch tensor).
    """
    fX = X / wx
    return fX

class MargLR_model_new_with_grid_search(nn.Module):
    def __init__(self, PA, X, param=None, noise_scale=np.sqrt(0.1)):
        super().__init__()
        self.param = param
        self.device = param["device"]
        self.Xinit = param.get('width_init')
        self.m = param.get('m', 200)

        self.mpa = None
        self.C = None
        self.D = None
        self.E = None
        self.F = None

        self.n = PA.shape[0]
        self.pa_list = param['pa_list']
        # kernel init
        PAkernel_width = self.MultiKernelInit_multi(PA, self.pa_list)
        Xkernel_width = self.Onekernel_median(X, self.Xinit)

        self.PAlength_scale_ = nn.Parameter(torch.tensor(PAkernel_width).to(self.device))
        self.PAlength_scale_.requires_grad = False

        self.amp = 1

        self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale).to(self.device))
        if param['optim'] == 'grid_search':
            self.noise_scale_.requires_grad = False
        else:
            self.noise_scale_.requires_grad = True
        

        self.Xlength_scale_ = torch.tensor(Xkernel_width).to(self.device)
        # self.Xlength_scale_.requires_grad = False
        self.score = 1e5

    def MultiKernelInit_multi(self, PA, pa_list):
        if PA.max() == PA.min():
            return [10.]
        n = len(pa_list)
        width_list = np.random.randn(n)
        for i in range(n):
            PA_sub = PA[:, pa_list[i]]
            PA_sub = np.mat(PA_sub.numpy())
            T = PA_sub.shape[0]
            GX = np.sum(np.multiply(PA_sub, PA_sub), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * PA_sub * PA_sub.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
            width = width * 2.5
            width_list[i] = width
        return width_list

    def Onekernel_median(self, X, weight=None):
        if weight is not None:
            return torch.tensor(weight).numpy()
        X = np.mat(X.numpy())
        T = X.shape[0]
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        width = width * 2.5  # kernel width
        return width

    def calculate(self, PA, X):

        X_width = self.Xlength_scale_.cpu().numpy()
        PA_width = self.PAlength_scale_.cpu().numpy()

        X = X.cpu().numpy()
        PA = PA.cpu().numpy()

        kerneld = DiagGaussianKernel()

        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)

        if PA.max() == PA.min():
            phi_PA = np.ones((PA.shape[0], 1))
        else:
            fPA = feat_gen(PA, PA_width)
            phi_PA = kernel_icd(fPA, kerneld, self.m)

        # induce matrix for X
        fX = feat_gen(X, X_width)
        phi_X = kernel_icd(fX, kerneld, self.m)
        # print("kkkkkkkkkkkk")

        self.mpa = phi_PA.shape[1]


        C = phi_X.T @ phi_X  # mx x mx
        D = phi_PA.T @ phi_PA  # mpa x mpa
        E = phi_X.T @ phi_PA  # mx x mpa
        F = E.T  # mpa x mx

        self.C = torch.tensor(C).to(self.device)
        self.D = torch.tensor(D).to(self.device)
        self.E = torch.tensor(E).to(self.device)
        self.F = torch.tensor(F).to(self.device)

    def fit(self, PA, X):

        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(0.001, 10)
        self.noise_scale_.data = noise_scale

        n = PA.shape[0]

        if self.mpa is None and self.C is None and self.D is None and self.E is None and self.F is None:
            self.calculate(PA, X)
        Noise = self.noise_scale_
        G = (1 / Noise) * self.D + torch.eye(self.mpa, device=self.device)
        B = torch.linalg.inv(Noise * G)
        R = torch.linalg.cholesky(G)

        first_part = - 0.5 * (1 / Noise) * torch.sum((self.C) ** 2) + 0.5 * (1 / Noise) * torch.sum(
            (self.E @ B).T * (self.F @ self.C))

        second_part = - n * torch.sum(torch.log(torch.diag(R))) + 0.5 * n * n * torch.log(1 / Noise)

        third_part = - n * n * 0.5 * np.log(2 * np.pi)

        marginal_likelihood = (first_part + second_part + third_part)

        self.score = -marginal_likelihood

        return marginal_likelihood
    
    def grid_search(self,PA,X,num_noises):
        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(0.001, 10)
        self.noise_scale_.data = noise_scale

        n = PA.shape[0]

        if self.mpa is None and self.C is None and self.D is None and self.E is None and self.F is None:
            self.calculate(PA, X)
        # Noise = self.noise_scale_

        Noise_values=torch.linspace(0.001, 10, num_noises)

        self.mx=self.C.shape[0]
        C_expanded = self.C.unsqueeze(0).expand(num_noises, self.mx, self.mx)  # [num_noises, mx, mx]
        D_expanded = self.D.unsqueeze(0).expand(num_noises, self.mpa, self.mpa)  # [num_noises, mpa, mpa]
        E_expanded = self.E.unsqueeze(0).expand(num_noises, self.mx, self.mpa)  # [num_noises, mx, mpa]
        F_expanded = self.F.unsqueeze(0).expand(num_noises, self.mpa, self.mx)  # [num_noises, mpa, mx]
        I_expanded = torch.eye(self.mpa, device=self.device).unsqueeze(0).expand(
            num_noises, self.mpa, self.mpa
        )  # [num_noises, mpa, mpa]


        n = PA.shape[0]

        inv_Noise = (1.0 / Noise_values).view(num_noises, 1, 1)  # [num_noises, 1, 1]
        G = inv_Noise * D_expanded + I_expanded  # [num_noises, mpa, mpa]

        B = torch.linalg.inv(Noise_values.view(num_noises, 1, 1) * G)  # [num_noises, mpa, mpa]
        R = torch.linalg.cholesky(G)  # [num_noises, mpa, mpa]

       
        c2_sum = C_expanded.square().sum(dim=[1, 2])  # [num_noises]

        first_term1 = -0.5 * (1.0 / Noise_values) * c2_sum  # [num_noises]

        # F_expanded: [num_noises, mpa, mx],  C_expanded: [num_noises, mx, mx]
        # => torch.bmm: (F_expanded, C_expanded) => [num_noises, mpa, mx]
        FC = torch.bmm(F_expanded, C_expanded)  # [num_noises, mpa, mx]

        # EB: [num_noises, mx, mpa]
        EB = torch.bmm(E_expanded, B)  # [num_noises, mx, mpa]


        EB_t = EB.transpose(1, 2)  # [num_noises, mpa, mx]

        mul_term = EB_t * FC  # [num_noises, mpa, mx]
        sum_mul = mul_term.sum(dim=[1, 2])  # [num_noises]

        first_term2 = 0.5 * (1.0 / Noise_values) * sum_mul  # [num_noises]

        first_part = first_term1 + first_term2  # [num_noises]

        # 5.2) second_part
        diag_R = torch.diagonal(R, dim1=1, dim2=2)  # [num_noises, mpa]
        sum_log_diag_R = torch.log(diag_R).sum(dim=1)  # [num_noises]

        # second_part = - n * sum_log_diag_R + 0.5 * n * n * log(1 / Noise)
        second_part = -n * sum_log_diag_R + 0.5 * n * n * torch.log(1.0 / Noise_values)  # [num_noises]

        # 5.3) third_part
        const_val = -0.5 * n * n * np.log(2.0 * np.pi)
        third_part = const_val * torch.ones_like(Noise_values)  # [num_noises]

        marginal_likelihood = first_part + second_part + third_part  # [num_noises]
        max_score,max_index=torch.max(marginal_likelihood,dim=0)
        # print(max_index)
        # print(Noise_values[max_index])
        self.score = -max_score  # [num_noises], 与 marginal_likelihood

        return max_score

        # Noise = self.noise_scale_
        # G = (1 / Noise) * self.D + torch.eye(self.mpa, device=self.device)
        # B = torch.linalg.inv(Noise * G)
        # R = torch.linalg.cholesky(G)
        #
        # print(G.shape)
        # print(B.shape)
        # print(R.shape)
        # print('----------')
        #
        # first_part = - 0.5 * (1 / Noise) * torch.sum((self.C) ** 2) + 0.5 * (1 / Noise) * torch.sum(
        #     (self.E @ B).T * (self.F @ self.C))
        #
        # second_part = - n * torch.sum(torch.log(torch.diag(R))) + 0.5 * n * n * torch.log(1 / Noise)
        #
        # third_part = - n * n * 0.5 * np.log(2 * np.pi)
        #
        # marginal_likelihood = (first_part + second_part + third_part)
        #
        #
        # self.score = -marginal_likelihood
        #
        # return marginal_likelihood
    def train_step(self, PA, x, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        # range
        opt.zero_grad()
        nll = -self.fit(PA, x)
        nll.backward()
        opt.step()
        return {
            "score": nll.item(),
        }
