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

    return G, ind


class FMIModel_grad_in_hand(nn.Module):
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
        self.PAlength_scale_.requires_grad = False

        self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale))
        self.noise_scale_.requires_grad = False

        self.Xlength_scale_ = nn.Parameter(torch.tensor(Xkernel_width))
        self.Xlength_scale_.requires_grad = False
        self.score = 1e5

        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        self.sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        self.distance_PA = self.pairwise_difference_tensor_transposed(PA) ** 2
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

    def pairwise_difference_tensor_transposed(self,X):
        """
        Compute the pairwise difference tensor for a given n x d matrix X and return it as d x n x n.

        Args:
            X (torch.Tensor): Input matrix of shape (n, d), where each row is a data point.

        Returns:
            torch.Tensor: Pairwise difference tensor of shape (d, n, n), where T[:, i, j] corresponds to X[i, :] - X[j, :].
        """
        # Expand X along new dimensions to compute pairwise differences
        X_expanded_1 = X.unsqueeze(1)  # Shape: (n, 1, d)
        X_expanded_2 = X.unsqueeze(0)  # Shape: (1, n, d)

        # Compute pairwise differences
        pairwise_diff = X_expanded_1 - X_expanded_2  # Broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)

        # Transpose to get shape (d, n, n)
        pairwise_diff = pairwise_diff.permute(2, 0, 1)  # Rearrange dimensions (n, n, d) -> (d, n, n)

        return pairwise_diff

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

    def cal_PAkernel(self, PA):
        # if self.PAlength_scale_.requires_grad:
        length_scale = self.PAlength_scale_.data
        length_scale = length_scale.clamp_(0.1, 10)
        self.PAlength_scale_.data = length_scale

        # if self.Xlength_scale_.requires_grad:
        # print("Xgrad: ", self.Xlength_scale_.grad)
        xlength_scale = self.Xlength_scale_.data
        xlength_scale = xlength_scale.clamp_(0.1, 10)
        self.Xlength_scale_.data = xlength_scale

        # if self.noise_scale_.requires_grad:
        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(0.001, 10)
        self.noise_scale_.data = noise_scale

        Kpa = self.PAkernel_multi(PA, self.pa_list)
        K = Kpa + self.noise_scale_ * torch.eye(PA.shape[0], device=self.device)
        K[K < 0] = 0
        return K

    #
    def Xkernel(self, X):
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        Kx = torch.exp(- 0.5 * sqdist / self.Xlength_scale_ ** 2)
        return Kx

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
    def Contional_get_grad(self,PA,X):
        length_scale = self.PAlength_scale_.data
        length_scale = length_scale.clamp_(0.1, 10)
        self.PAlength_scale_.data = length_scale

        # if self.Xlength_scale_.requires_grad:
        # print("Xgrad: ", self.Xlength_scale_.grad)
        xlength_scale = self.Xlength_scale_.data
        xlength_scale = xlength_scale.clamp_(0.1, 10)
        self.Xlength_scale_.data = xlength_scale

        # if self.noise_scale_.requires_grad:
        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(0.001, 10)
        self.noise_scale_.data = noise_scale

        # print("wPA: ", self.PAlength_scale_, "wX: ", self.Xlength_scale_, "Noise: ",  self.noise_scale_)

        # X_np = X.cpu().detach().numpy()
        # PA_np = PA.cpu().detach().numpy()

        X_width = self.Xlength_scale_
        PA_width = self.PAlength_scale_
        Noise = self.noise_scale_
        # print(X_width, PA_width, Noise)
        m = self.m
        n = X.shape[0]

        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)
        start_time = time.time()

        fX = feat_gen(X, X_width)
        kerneld = DiagGaussianKernel_torch_update()
        G_X, ind_X = kernel_icd_torch_update(fX, kerneld, m)
        phi_X = G_X

        # induce matrix for PA
        if PA.max() == PA.min():
            phi_PA = torch.ones((n, 1), dtype=torch.float64, device=X.device)
        else:
            fPA = feat_gen(PA, PA_width)
            kerneld = DiagGaussianKernel_torch_update()
            G_PA, ind_PA = kernel_icd_torch_update(fPA, kerneld, m)
            phi_PA = G_PA

  

        mpa = phi_PA.shape[1]


        # phi_PAc = torch.tensor(phi_PA).to(self.device)

        # print(phi_Xc.type(), phi_PAc.type())
        # print(phi_Xc.shape, phi_PAc.shape)

        C = phi_X.T @ phi_X  # mx x mx
        D = phi_PA.T @ phi_PA  # mpa x mpa
        E = phi_X.T @ phi_PA  # mx x mpa
        F = E.T  # mpa x mx

        # C = torch.tensor(C).to(self.device)
        # D = torch.tensor(D).to(self.device)
        # E = torch.tensor(E).to(self.device)
        # F = torch.tensor(F).to(self.device)

        G = (1 / Noise) * D + torch.eye(mpa, device=self.device)
        # print(phi_Xc, phi_PAc)
        B = torch.linalg.inv(Noise * G)
        R = torch.linalg.cholesky(G)

        # Xsq = (X ** 2).sum(dim=1, keepdim=True)
        # sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        sqdist=self.sqdist.to(self.device)
        Kx = torch.exp(- 0.5 * sqdist / self.Xlength_scale_ ** 2)
        grad_k_widx = ((sqdist / (self.Xlength_scale_ ** 3)) * Kx) @ (phi_X - phi_PA @ B @ F)
        final_grad_X = (1 / self.noise_scale_) * torch.trace(phi_X.T @ grad_k_widx) / (n * n)
        # print(f'final_X_grad:{final_grad_X}')



        # final_grad_Noise = -n*((0.5 * 1 / self.noise_scale_) * (n-torch.trace(B @ D)) + 1 / (
        #             2 * self.noise_scale_**2) * torch.trace(C @ (C - 2 * E @ B @ F + F.T @ B @ D @ B @ F))) / (n * n)
        final_grad_Noise1 = n * ((0.5 * 1 / self.noise_scale_) * (n - torch.trace(B @ D))) / (n * n)
        final_grad_Noise2 = -(
                    1 / (2 * self.noise_scale_ ** 2) * torch.trace(C @ (C - 2 * E @ B @ F + F.T @ B @ D @ B @ F))) / (n * n)
        final_grad_Noise = final_grad_Noise1 + final_grad_Noise2
        # print(f'grad_N:{final_grad_Noise}')

        # distance_PA = self.pairwise_difference_tensor_transposed(PA) ** 2
        distance_PA=self.distance_PA.to(self.device)
        PA_width = PA_width.reshape(-1, 1, 1)
        Kpa = torch.exp(-torch.sum(0.5 * distance_PA / PA_width ** 2, dim=0))
        K_grad = 1 / (PA_width ** 3) * distance_PA * Kpa

        H = phi_X.T - E @ B @ phi_PA.T
        I = C @ H

        I_expanded = I.unsqueeze(0)  # Shape: (1, 8, 200)
        I_expanded = I_expanded.expand(K_grad.shape[0], -1, -1)  # Shape: (d, 8, 200)


        I2 = torch.matmul(I_expanded, K_grad)  # Shape: (d, 8, 200)
        H2 = H.T
        H2 = H2.unsqueeze(0)
        H2 = H2.expand(K_grad.shape[0], -1, -1)

        I3 = torch.matmul(I2, H2)
        diagonals = torch.diagonal(I3, dim1=1, dim2=2)  # Shape: (d, n)
        # Sum over the diagonal elements to compute the trace
        traces = diagonals.sum(dim=1)  # Shape: (d,)
        final_pa_grad1 = 1 / (2 * self.noise_scale_ ** 2) * traces / (n * n)

        diagonals_part2 = torch.diagonal(K_grad, dim1=1, dim2=2)  # Shape: (d, n)
        # Sum over the diagonal elements to compute the trace
        traces2 = diagonals_part2.sum(dim=1)  # Shape: (d,)

        P = B @ phi_PA.T

        P_expanded = P.unsqueeze(0)  # Shape: (1, 8, 200)
        P_expanded = P_expanded.expand(K_grad.shape[0], -1, -1)  # Shape: (d, 8, 200)


        L2 = torch.matmul(P_expanded, K_grad)  # Shape: (d, 8, 200)
        P2 = P.T
        P2 = P2.unsqueeze(0)
        P2 = P2.expand(K_grad.shape[0], -1, -1)

        I3 = torch.matmul(L2, P2)
        diagonals3 = torch.diagonal(I3, dim1=1, dim2=2)  # Shape: (d, n)
        # Sum over the diagonal elements to compute the trace
        traces3 = diagonals3.sum(dim=1)  # Shape: (d,)
        final_pa_grad2 = n / (2 * self.noise_scale_) * (traces2 - traces3) / 2 / (n * n)
        final_pa_grad = -(final_pa_grad2 + final_pa_grad1)
        #print(f'final_pa_grad:{final_pa_grad}')
        return final_grad_X,final_grad_Noise,final_pa_grad

    def ConditionalLikelihood(self, PA, X):
        # if self.PAlength_scale_.requires_grad:
        length_scale = self.PAlength_scale_.data
        length_scale = length_scale.clamp_(0.1, 10)
        self.PAlength_scale_.data = length_scale

        # if self.Xlength_scale_.requires_grad:
        # print("Xgrad: ", self.Xlength_scale_.grad)
        xlength_scale = self.Xlength_scale_.data
        xlength_scale = xlength_scale.clamp_(0.1, 10)
        self.Xlength_scale_.data = xlength_scale

        # if self.noise_scale_.requires_grad:
        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(0.001, 10)
        self.noise_scale_.data = noise_scale

        # print("wPA: ", self.PAlength_scale_, "wX: ", self.Xlength_scale_, "Noise: ",  self.noise_scale_)

        # X_np = X.cpu().detach().numpy()
        # PA_np = PA.cpu().detach().numpy()

        X_width = self.Xlength_scale_
        PA_width = self.PAlength_scale_
        Noise = self.noise_scale_
        # print(X_width, PA_width, Noise)
        m = self.m
        n = X.shape[0]

        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)
        start_time = time.time()

        fX = feat_gen(X, X_width)
        kerneld = DiagGaussianKernel_torch_update()
        G_X, ind_X = kernel_icd_torch_update(fX, kerneld, m)
        phi_X = G_X

        # induce matrix for PA
        if PA.max() == PA.min():
            phi_PA = torch.ones((n, 1), dtype=torch.float64, device=X.device)
        else:
            fPA = feat_gen(PA, PA_width)
            kerneld = DiagGaussianKernel_torch_update()
            G_PA, ind_PA = kernel_icd_torch_update(fPA, kerneld, m)
            phi_PA = G_PA


        mpa = phi_PA.shape[1]

        # phi_PAc = torch.tensor(phi_PA).to(self.device)

        # print(phi_Xc.type(), phi_PAc.type())
        # print(phi_Xc.shape, phi_PAc.shape)

        C = phi_X.T @ phi_X  # mx x mx
        D = phi_PA.T @ phi_PA  # mpa x mpa
        E = phi_X.T @ phi_PA  # mx x mpa
        F = E.T  # mpa x mx

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

        # marginal_likelihood=second_part

        # Xsq = (X ** 2).sum(dim=1, keepdim=True)
        # sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        # Kx = torch.exp(- 0.5 * sqdist / self.Xlength_scale_ ** 2)
        # grad_k_widx = ((sqdist / (self.Xlength_scale_ ** 3)) * Kx) @ (phi_Xc - phi_PAc @ B @ F)
        # final_grad_X = (1 / self.noise_scale_) * torch.trace(phi_Xc.T @ grad_k_widx) / (n * n)
        # # print(f'grad_X:{final_grad_X}')
        #
        # # final_grad_Noise = -n*((0.5 * 1 / self.noise_scale_) * (n-torch.trace(B @ D)) + 1 / (
        # #             2 * self.noise_scale_**2) * torch.trace(C @ (C - 2 * E @ B @ F + F.T @ B @ D @ B @ F))) / (n * n)
        # final_grad_Noise1 = n*((0.5 * 1 / self.noise_scale_) * (n-torch.trace(B @ D)))/(n * n)
        # final_grad_Noise2=-(1 / (2 * self.noise_scale_**2) * torch.trace(C @ (C - 2 * E @ B @ F + F.T @ B @ D @ B @ F))) / (n * n)
        # final_grad_Noise=final_grad_Noise1+final_grad_Noise2
        # print(f'grad_N:{final_grad_Noise}')
        #
        #
        # distance_PA=self.pairwise_difference_tensor_transposed(PA)**2
        # PA_width=PA_width.reshape(-1,1,1)
        # Kpa=torch.exp(-torch.sum(0.5*distance_PA/PA_width**2,dim=0))
        # K_grad=1/(PA_width**3)*distance_PA*Kpa
        #
        #
        #
        # I3=torch.matmul(I2,H2)
        # diagonals = torch.diagonal(I3, dim1=1, dim2=2)  # Shape: (d, n)
        # # Sum over the diagonal elements to compute the trace
        # traces = diagonals.sum(dim=1)  # Shape: (d,)
        # final_pa_grad1=1/(2*self.noise_scale_**2)*traces/(n*n)
        #
        # diagonals_part2 = torch.diagonal(K_grad, dim1=1, dim2=2)  # Shape: (d, n)
        # # Sum over the diagonal elements to compute the trace
        # traces2= diagonals_part2.sum(dim=1)  # Shape: (d,)
        #
        #
        #
        # I3=torch.matmul(L2,P2)
        # diagonals3 = torch.diagonal(I3, dim1=1, dim2=2)  # Shape: (d, n)
        # # Sum over the diagonal elements to compute the trace
        # traces3 = diagonals3.sum(dim=1)  # Shape: (d,)
        # final_pa_grad2=n/(2*self.noise_scale_)*(traces2-traces3)/2/(n*n)
        # final_pa_grad=-(final_pa_grad2+final_pa_grad1)
        # print(f'final_pa_grad:{final_pa_grad}')
        # print(f'score:{marginal_likelihood}')

        return marginal_likelihood/n


        return driv_score / n

    def fast_Jacobian_gradx(self,X):
        n=X.shape[0]
        term1 = (n * torch.sum(X ** 2) - (torch.sum(X)) ** 2)
        X_grad=-2*n*(n-1)*(1/self.Xlength_scale_)+(2/self.Xlength_scale_**3)*term1
        return X_grad/(n*n)

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
        return (term1 + term2) / n


    def Score(self, x, y,flag):
        """
        score function using marginal likelihood of joint distribution
        :param x: independent variable. N x m
        :param y: dependent variable. N x 1
        :return:
        """
        n = x.shape[0]
        x = x.to(self.device)
        y = y.to(self.device)
        start_time1 = time.time()
        if flag==True:
            marginal_likelihood = self.ConditionalLikelihood(x, y) / n
        else:
            marginal_likelihood=0
        grad_x1,grad_n,grad_pa=self.Contional_get_grad(x,y)
        end_time1 = time.time()
        run_time1 = end_time1 - start_time1
        # print('marg_time', run_time1)

        start_time2 = time.time()
        # driv_term1 = self.Jacobian(y) / n
        # driv_term2 = self.multi_Jacobian(y) / n
        if flag==True:
            driv_term3 = self.fast_Jacobian(y) / n
        grad_x2=self.fast_Jacobian_gradx(y)
        grad_x=grad_x1-grad_x2

        # print(f'grad_X:{grad_x}')
        # print(f'grad_PA:{grad_pa}')
        if flag == True:
            driv_term = driv_term3
        else:
            driv_term=0
        end_time2 = time.time()
        run_time2 = end_time2 - start_time2
        # print('jac_time', run_time2)
        if flag==True:
            nll = -(marginal_likelihood + driv_term)
        else:
            nll=torch.tensor(0)
        # nll = -marginal_likelihood

        self.score = nll.cpu().detach().numpy()
        return nll, marginal_likelihood, driv_term,grad_x,grad_n,grad_pa

    def grad_in_hand(self):
        pass

    def train_step(self, x, y, opt,flag):
        opt.zero_grad()
        start = time.time()

        score, nlml, Jacobian,grad_x,grad_n,grad_pa = self.Score(x, y,flag)
        # score.backward()
        end = time.time()
        #print(f"time:{end - start}")
        #print("Xgrad:", self.Xlength_scale_.grad)
        # print("PAgrad",self.PAlength_scale_.grad)
        # print("noise_scale",self.noise_scale_)
        # print("Score:", score)
        self.Xlength_scale_.requires_grad=True
        self.noise_scale_.requires_grad=True
        self.PAlength_scale_.requires_grad=True
        self.Xlength_scale_.grad=grad_x.to(self.Xlength_scale_.dtype)
        self.noise_scale_.grad=grad_n
        if grad_pa.dtype != self.PAlength_scale_.dtype:
            grad_pa = grad_pa.to(self.PAlength_scale_.dtype)
        self.PAlength_scale_.grad = grad_pa
        opt.step()
        self.Xlength_scale_.requires_grad = False
        self.noise_scale_.requires_grad = False
        self.PAlength_scale_.requires_grad = False
        if flag==True:
            return {'score': score.item(),
                    'nlml': nlml.detach().cpu(),
                    'driv': Jacobian.detach().cpu(),
                    'PAlength': self.PAlength_scale_.data,
                    'Xlength': self.Xlength_scale_.data,
                    }
        else:
            return {'score': score.item(),
                    'PAlength': self.PAlength_scale_.data,
                    'Xlength': self.Xlength_scale_.data,
                    }

