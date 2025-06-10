from .moe import MoE
from typing import Any, Dict, Tuple, List, Union, Optional
from .register import register_moe
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
from framework.layers import LoggingLayer
from framework.layers import RegularizedLayer
from framework import utils
import framework
import math
from framework.layers import OncePerIterLayer
from layers import cvmm, cvmm_prepare_sel
from layers.cvmm import CVMMSel, cvmm_prepare_sel2
import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F
import copy
import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.layers import RegularizedLayer
from collections import namedtuple
Selection = namedtuple('Selection', ['raw_sel', 'sel_val', 'raw_sel_index', 'sel_index'])

@register_moe("xmoe")
class XMOE(MoE):
    def __init__(self, 
                 dmodel: int, 
                 n_experts: int, 
                 expert_size: int, 
                 n_heads: int, 
                 std_gate = float,
                 std_expert = float,
                 topk=2,
                 dropout: float = 0, weight_scale: float = 1.0,
                 selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
                 perplexity_reg_mode: str = "step",
                 activation_after_topk: bool = False,
                 activation=lambda x: F.relu(x, inplace=True),
                 sel_bias: bool = False,
                 bias: bool = False,
                 v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 sync_distributed: bool = False,
                 selection_dropout: float = 0.0,
                 log_interval: Optional[int] = 100,
                 args=None,
                 std = 1,
                 out_dmodel = None,
                 is_att = False,
                inp_expert = None, # define dim for o linear when attention. so gate competed previous and dim input gate and expert different
                out_expert = None
                
                 ):

        # Gọi constructor của lớp cha MoeLayer với đầy đủ các tham số
        super().__init__(dmodel=dmodel, 
                         n_experts=n_experts, 
                         expert_size=expert_size, 
                         n_heads=n_heads, 
                         topk=topk,
                         dropout=dropout, 
                         weight_scale=weight_scale, 
                         selection_mode=selection_mode, 
                         perplexity_reg=perplexity_reg,
                         perplexity_reg_mode=perplexity_reg_mode, 
                         activation_after_topk=activation_after_topk, 
                         activation=activation,
                         sel_bias=sel_bias, 
                         bias=bias, 
                         v_dim=v_dim, 
                         expert_dropout=expert_dropout, 
                         sync_distributed=sync_distributed,
                         selection_dropout=selection_dropout, 
                         log_interval=log_interval, 
                         args=args,
                         out_dmodel = out_dmodel,
                         is_att = is_att,
                         out_expert = out_expert,
                         inp_expert = inp_expert,
                         std_gate = std_gate,
                         std_expert = std_expert,
                         )        
        self.reduction_dim = int(self.n_experts / 2)
        expert_embeddings = torch.empty(self.num_of_experts, int(self.reduction_dim))
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)         
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )
        
        # self.inp_reduction = torch.nn.Linear(self.k_dim, int(self.num_of_experts / 2), bias=False)
        self.temperature = 0.3
        self.bias = None
        
        self.expert_sel = torch.nn.Parameter(torch.empty(self.reduction_dim, self.k_vec_dim))
        self.sel_bias = torch.nn.Parameter(torch.zeros(self.reduction_dim)) if sel_bias else None
        self.get_initializer()(self.expert_sel, std=self.k_vec_dim ** -0.5 * self.sel_weight_scale)
        # input embedding
        # self.inp_reduction = torch.nn.Linear(self.k_vec_dim, self.reduction_dim, bias=False)
        self.inp_reduction = lambda x: F.linear(x, self.expert_sel, self.sel_bias)
    def _keepTopk(self, x):
        weights, selected_experts  = torch.topk(x, k = self.num_selected, dim=2) 
        weights = torch.softmax(weights, dim=-1)
        return weights, selected_experts 
    def _cosine(self, mat1, mat2, eps=1e-4):
        """
        Compute the cosine similarity between mat1 and mat2.

        Args:
            mat1 (torch.Tensor): Input tensor of shape (B, N, D')
            mat2 (torch.Tensor): Expert embeddings of shape (E, D')
            eps (float): Small value to avoid division by zero

        Returns:
            torch.Tensor: Cosine similarity scores of shape (B, N, E)
        """
        # Normalize mat1 across the last dimension (D')
        mat1_normalized = F.normalize(mat1.float(), p=2.0, dim=-1, eps=eps)

        # Compute cosine similarity: (B, N, D') @ (D', E) -> (B, N, E)
        cosine_similarity = torch.matmul(mat1_normalized, mat2.float().transpose(0, 1))
        
        return cosine_similarity.type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
    def compute_gate(self, x):
        reduced_inp = self.inp_reduction(x)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=-1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
            
        gate_logits = self._cosine(reduced_inp, self.expert_embeddings)
        gate_logits = self._make_finite(gate_logits)
    
        return gate_logits
    # def compute_gate(self, x)
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        # compute output
        in1 = in2 = x
        
        gate_logits = self.compute_gate(x)
        gate_softmax = F.softmax(gate_logits / self.temperature, dim=-1, dtype=torch.float).to(x.dtype)
        weights, selected_experts = self._keepTopk(gate_softmax)
    
        sel_indices = cvmm_prepare_sel2(selected_experts.int())
        scores = self.compute_scores(in2, sel_indices)
        
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = weights
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None
        
        out = cvmm(scores, sel_indices, self.values)

        self.layer += 1

        self.was_training = self.training
        res = out.view(*x.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        
        name = f"{self.name_moe}_ebalance"
        bal_loss = self.entropy_balance(gate_logits)* (self.args.balance_loss_coef / self.div)
        self.add_reg(lambda: bal_loss, name)
        if self.args.test_only:
            self.add_dist_experts(selection=selected_experts)
            self.add_dist_weight(weight=weights)
            self.add_dist_weight(weight=gate_softmax, is_all=True)
        return res
        # compute loss
        # if self.training:
        #     self.update_aux_statistics(
        #             gate_logits=gate_logits,
        #             gate_softmax=gate_softmax,
        #             selected_experts=selected_experts
        #         )
        
        # return res
    
    # def att_forward(self, x, n_experts, n_copies, return_full = True, *args, **kwargs):
        
    #     if self.selection_dropout > 0 and self.training:
    #         x = F.dropout(x, self.selection_dropout)

    #     gate_logits = self.compute_gate(x)
    #     gate_logits = gate_logits.view(*gate_logits.shape[:-1], n_copies, -1)
    #     gate_softmax = F.softmax(gate_logits / self.temperature, dim=-1, dtype=torch.float).to(x.dtype)
    #     with torch.no_grad():
    #         _, gate_softmax_index = gate_softmax.topk(self.num_selected, dim=-1, sorted=False)
            
    #     gate_logits_val = torch.gather(gate_softmax, -1, gate_softmax_index)
    #     gate_logits_val = torch.softmax(gate_logits_val, dim=-1)
        
    #     gate_logits_index_shifted = (torch.arange(n_copies, device=gate_softmax_index.device, dtype=gate_softmax_index.dtype) * n_experts).unsqueeze(-1) + gate_softmax_index
        
    #     gate_logits_index_pp = cvmm_prepare_sel2(gate_logits_index_shifted.flatten(-2,-1).int(), gate_logits_val)
    #     # compute loss
    #     if self.training:
    #         self.update_aux_statistics(
    #                 gate_logits=gate_logits,
    #                 gate_softmax=gate_softmax,
    #                 selected_experts=gate_softmax_index
    #             )
    #     return Selection(gate_logits, gate_logits_val, gate_softmax_index, gate_logits_index_pp)

    # def compute_moe(self, x: torch.Tensor, sel: Selection) -> torch.Tensor:
    #     return cvmm(x, sel.sel_index, self.experts)
