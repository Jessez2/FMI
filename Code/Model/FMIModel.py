import numpy as np
import torch
import torch.nn as nn
import warnings
import time


def feat_gen(X, wx):
    """
    Generates features by dividing the data X by the vector wx.

    :param X: Input data matrix (torch tensor).
    :param wx: Feature scaling vector (torch tensor).
    :return: Scaled feature matrix (torch tensor).
    """
    fX = X / wx
    return fX


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


class DiagGaussianKernel_torch_update:
    """
    A class for a simplified diagonal version of the Gaussian Kernel.
    """

    def __init__(self, sigma=1.0):
        """
        Initialise object with the given value of sigma >= 0.

        :param sigma: Kernel width parameter.
        :type sigma: float
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Computes the diagonal Gaussian kernel between X1 and X2.

        :param X1: First set of examples (torch tensor).
        :param X2: Second set of examples (torch tensor).
        :return: Kernel values between X1 and X2 (torch tensor).
        """
        K = torch.exp(- torch.sum((X1 - X2) ** 2, dim=1) / (2 * self.sigma ** 2))
        K = K.view(-1, 1)  # Convert to column vector (torch tensor)
        return K


def kernel_icd_torch_update(X, kernel, m=100, precision=1e-6):
    n = X.shape[0]

    perm = np.arange(n)  # permutation vector
    d = torch.zeros((1, n), dtype=torch.float64, device=X.device)  # diagonal of the residual kernel matrix
    G = torch.zeros((n, m), dtype=torch.float64, device=X.device)
    for i in range(m):
        x = X[perm[i:n + 1], :]  # Fixed slicing to avoid n + 1

        if i == 0:  # diagonal of kernel matrix
            d_new = kernel(x, x).T
        else:  # update the diagonal of the residual kernel matrix
            residual = torch.sum(G[i:n + 1, :i] ** 2, dim=1)
            d_new = kernel(x, x).squeeze() - residual

        if i == 0:
            d = d_new
        else:
            d = torch.cat([d[:, :i], d_new.unsqueeze(0)], dim=1)

        dtrace = torch.sum(d[:, i:n + 1])

        if dtrace <= precision:  # mark
            G = G[:, :i]
            # G = G[:, :i].clone()
            break

        start_time_G_new = time.time()
        m2 = torch.max(d[:, i:n + 1])  # mark
        j = torch.argmax(d[:, i:n + 1]).item()
        j = j + i  # take into account the offset i
        m1 = torch.sqrt(m2)


        perm[[i, j]] = perm[[j, i]]

        G_new = G.clone()

        temp_G_new_ji = G_new[j, :i].clone()
        G_new[j, :i] = G_new[i, :i]
        G_new[i, :i] = temp_G_new_ji
        G_new[i, i] = m1

        # Calculate the i-th column without in-place modification
        kernel_vals = kernel(X[perm[i], :], X[perm[i + 1:n + 1], :]).squeeze()  # Shape: [n-i]
        adjustment = torch.matmul(G_new[i + 1:n + 1, :i], G_new[i, :i].T)  # Shape: [n-i]

        G_new_column = (kernel_vals - adjustment) / m1

        G_new = G_new.clone()

        G_new[i + 1:n + 1, i] = G_new_column

        G = G_new

    ind = np.argsort(perm)

    G = G[ind, :]

    return G


class FMIModel(nn.Module):
    def __init__(self, PA, X, param, noise_scale=np.sqrt(0.1)):
        super().__init__()
        self.param = param
        self.device = param['device']
        self.Xinit = param.get('width_init')
        self.m = param.get('m', 200)

        self.n = PA.shape[0]
        self.pa_list = param['pa_list']
        self.threshold = param['threshold']
        self.amp = 1

        PAkernel_width = self.MultiKernelInit_multi(PA, self.pa_list)
        Xkernel_width = self.Onekernel_median(X, self.Xinit)

        self.PAlength_scale_ = nn.Parameter(torch.tensor(PAkernel_width))
        self.PAlength_scale_.requires_grad = True

        self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale))
        self.noise_scale_.requires_grad = True

        self.Xlength_scale_ = nn.Parameter(torch.tensor(Xkernel_width))
        self.Xlength_scale_.requires_grad = True
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


    def PAkernel_multi(self, PA, pa_list):
        n = len(pa_list)
        for i in range(n):
            PAt = PA[:, pa_list[i]]
            PAt = PAt / self.PAlength_scale_[i]
            PAtsq = (PAt ** 2).sum(dim=1, keepdim=True)
            dist_i = PAtsq + PAtsq.T - 2 * PAt.mm(PAt.T)
            if i == 0:
                sqdist = dist_i
            else:
                sqdist += dist_i

        Kpa = torch.exp(-0.5 * sqdist)
        return Kpa

    def ConditionalLikelihood(self, PA, X):
        if self.PAlength_scale_.requires_grad:
            length_scale = self.PAlength_scale_.data
            length_scale = length_scale.clamp_(0.1, 10)
            self.PAlength_scale_.data = length_scale

        if self.Xlength_scale_.requires_grad:
            # print("Xgrad: ", self.Xlength_scale_.grad)
            xlength_scale = self.Xlength_scale_.data
            xlength_scale = xlength_scale.clamp_(0.1, 10)
            self.Xlength_scale_.data = xlength_scale

        if self.noise_scale_.requires_grad:
            noise_scale = self.noise_scale_.data
            noise_scale = noise_scale.clamp_(0.001, 10)
            self.noise_scale_.data = noise_scale

        # print("wPA: ", self.PAlength_scale_, "wX: ", self.Xlength_scale_, "Noise: ",  self.noise_scale_)

        # X_np = X.cpu().detach().numpy()
        # PA_np = PA.cpu().detach().numpy()

        X_width = self.Xlength_scale_
        PA_width = self.PAlength_scale_
        Noise = self.noise_scale_

        m = self.m
        n = X.shape[0]

        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)

        fX = feat_gen(X, X_width)
        kerneld = DiagGaussianKernel_torch_update()
        G_X = kernel_icd_torch_update(fX, kerneld, m)
        phi_X = G_X

        # induce matrix for PA
        if PA.max() == PA.min():
            phi_PA = torch.ones((n, 1), dtype=torch.float64, device=X.device)
        else:
            fPA = feat_gen(PA, PA_width)
            kerneld = DiagGaussianKernel_torch_update()
            G_PA = kernel_icd_torch_update(fPA, kerneld, m)
            phi_PA = G_PA
        # print("gxphipa: ", G_PA, phi_PA)

        # fPA = feat_gen(PA, PA_width_np)
        # kerneld = DiagGaussianKernel_numpy()
        # phi_PA, _ = kernel_icd_numpy(fPA.cpu().detach().numpy(), kerneld, m)


        mpa = phi_PA.shape[1]

        C = phi_X.T @ phi_X  # mx x mx
        D = phi_PA.T @ phi_PA  # mpa x mpa
        E = phi_X.T @ phi_PA  # mx x mpa
        F = E.T

        # C = torch.tensor(C).to(self.device)
        # D = torch.tensor(D).to(self.device)
        # E = torch.tensor(E).to(self.device)
        # F = torch.tensor(F).to(self.device)

        G = (1 / Noise) * D + torch.eye(mpa, device=self.device)
        B = torch.linalg.inv(Noise * G)
        R = torch.linalg.cholesky(G)

        first_part = - 0.5 * (1 / Noise) * torch.sum(C ** 2) + 0.5 * (1 / Noise) * torch.sum((E @ B).T * (F @ C))

        second_part = - n * torch.sum(torch.log(torch.diag(R))) + 0.5 * n * n * torch.log(1 / Noise)

        third_part = - n * n * 0.5 * np.log(2 * np.pi)

        marginal_likelihood = first_part + second_part + third_part

        # print('score', marginal_likelihood)

        return marginal_likelihood / n

    def fast_Jacobian(self, X):

        n = X.shape[0]
        Xlength_scale_ = self.Xlength_scale_
        term1 = -(n * torch.sum(X ** 2) - (torch.sum(X)) ** 2) / Xlength_scale_ ** 2
        # print(term1)
        _, num = torch.unique(X, return_counts=True)
        num_no_zero = torch.sum((torch.sum(num) - num) * num)
        term2 = -2 * num_no_zero * torch.log(Xlength_scale_)

        xm = X.T.unsqueeze(2)
        Xm = X.T.unsqueeze(1)
        dist = (xm - Xm)[0]
        # print(dist)
        abs_deriv_diag = torch.abs(dist)
        # print(abs_deriv_diag)
        nonZerosIdx = abs_deriv_diag > self.threshold
        # print(torch.sum(nonZerosIdx), n*(n-1))

        log_deriv_diag = torch.log(abs_deriv_diag[nonZerosIdx])
        # print(log_deriv_diag)
        term3 = torch.sum(log_deriv_diag)

        return (term1 + term2 + term3) / n

    def Score(self, x, y):
        n = x.shape[0]
        x = x.to(self.device)
        y = y.to(self.device)
        start_time = time.time()
        driv_term3 = self.fast_Jacobian(y) / n
        marginal_likelihood = self.ConditionalLikelihood(x, y) / n
        driv_term = driv_term3
        nll = -(marginal_likelihood + driv_term)
        end_time = time.time()
        run_time = end_time - start_time
        self.score = nll.cpu().detach().numpy()
        return nll, marginal_likelihood, driv_term

    def train_step(self, x, y, opt):
        opt.zero_grad()
        score, nlml, Jacobian = self.Score(x, y)
        score.backward()
        #print("Xgrad:", self.Xlength_scale_.grad)
        opt.step()
        return {'score': score.item(),
                'nlml': nlml.detach().cpu(),
                'driv': Jacobian.detach().cpu(),
                'PAlength': self.PAlength_scale_.data,
                'Xlength': self.Xlength_scale_.data,
                }
