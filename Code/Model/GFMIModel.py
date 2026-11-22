import argparse
import ast
import difflib
import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path

import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(module, x, pa, rank):
    obj = module.FMIModel_grad_in_hand.__new__(module.FMIModel_grad_in_hand)
    torch.nn.Module.__init__(obj)
    obj.m, obj.device = rank, "cpu"
    obj.Xlength_scale_ = torch.nn.Parameter(x.new_tensor(1.1), requires_grad=False)
    obj.PAlength_scale_ = torch.nn.Parameter(x.new_full((pa.shape[1],), 1.3), requires_grad=False)
    obj.noise_scale_ = torch.nn.Parameter(x.new_tensor(0.4), requires_grad=False)
    obj.sqdist = (x.square().sum(1, keepdim=True) + x.square().sum(1)[None, :] - 2 * x @ x.T)
    obj.distance_PA = obj.pairwise_difference_tensor_transposed(pa).square()
    return obj


def verify(original_path, corrected_path):
    original = original_path.read_text(encoding="utf-8-sig").splitlines()
    corrected = corrected_path.read_text(encoding="utf-8-sig").splitlines()
    replacements = {
        "        P2 = P.T": "        P2 = phi_PA",
        "        final_pa_grad2 = n / (2 * self.noise_scale_) * (traces2 - traces3) / 2 / (n * n)":
        "        final_pa_grad2 = n / (2 * self.noise_scale_) * (traces2 - traces3) / (n * n)",
        "        final_pa_grad = -(final_pa_grad2 + final_pa_grad1)":
        "        final_pa_grad = final_pa_grad2 - final_pa_grad1",
    }
    assert [replacements.get(line, line) for line in original] == corrected
    ast.parse("\n".join(corrected))
    old, new = load(original_path, "original_pa_model"), load(corrected_path, "corrected_pa_model")
    torch.set_num_threads(2)
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = torch.tensor(rng.normal(size=(40, 1)), dtype=torch.float64)
        for dimensions in (0, 1, 3):
            pa = torch.tensor(rng.normal(size=(40, max(1, dimensions))), dtype=torch.float64)
            if dimensions == 0:
                pa.zero_()
            for rank in (8, 40):
                a, b = model(old, x, pa, rank), model(new, x, pa, rank)
                old_grads = a.Contional_get_grad(pa, x)
                factors = []
                selector = new.kernel_icd_torch_update

                def capture(*args, **kwargs):
                    result = selector(*args, **kwargs)
                    factors.append(result[0].detach())
