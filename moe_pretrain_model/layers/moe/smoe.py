from .moe import MoE
from typing import Any, Dict, Tuple, List, Union, Optional
from .register import register_moe
import torch
import torch.nn as nn
import torch.nn.functional as F
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

@register_moe("smoe")
class SMoeLayer(MoE):
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
        # self.activation_functions =[
        #     (lambda x: torch.mean(F.relu(x), dim=-1), 'Mean ReLU'),
        #     (lambda x: torch.mean(F.leaky_relu(x), dim=-1), 'Mean Leaky ReLU'),
        #     (lambda x: torch.mean(F.prelu(x, torch.tensor(0.25, device=x.device)), dim=-1), 'Mean PReLU'),
        #     (lambda x: torch.mean(F.elu(x), dim=-1), 'Mean ELU'),
        #     (lambda x: torch.mean(F.selu(x), dim=-1), 'Mean SELU'),
        #     (lambda x: torch.mean(F.gelu(x), dim=-1), 'Mean GELU'),
        #     (lambda x: torch.mean(F.silu(x), dim=-1), 'Mean SiLU (Swish)'),
        #     (lambda x: torch.mean(F.tanh(x), dim=-1), 'Mean Tanh'),
        #     (lambda x: torch.mean(F.sigmoid(x), dim=-1), 'Mean Sigmoid'),
        #     (lambda x: torch.mean(F.softplus(x), dim=-1), 'Mean Softplus'),
        #     (lambda x: torch.mean(F.softsign(x), dim=-1), 'Mean Softsign'),
        #     (lambda x: torch.mean(F.hardswish(x), dim=-1), 'Mean HardSwish'),
        #     (lambda x: torch.mean(F.hardtanh(x), dim=-1), 'Mean HardTanh'),
        #     (lambda x: torch.mean(F.threshold(x, 0, 0), dim=-1), 'Mean Threshold'),
        #     (lambda x: torch.mean(F.tanhshrink(x), dim=-1), 'Mean Tanhshrink'),
        #     (lambda x: torch.mean(F.logsigmoid(x), dim=-1), 'Mean LogSigmoid'),
        #     (lambda x: torch.mean(F.log_softmax(x, dim=-1), dim=-1), 'Mean LogSoftmax'),
        #     (lambda x: torch.mean(F.softmax(x, dim=-1), dim=-1), 'Mean Softmax'),
        #     (lambda x: torch.mean(F.gelu(x), dim=-1), 'Mean GELU'),
        #     (lambda x: torch.mean(F.selu(x), dim=-1), 'Mean SELU'),
        #     (lambda x: torch.mean(F.silu(x), dim=-1), 'Mean SiLU'),
        #     (lambda x: torch.mean(F.leaky_relu(x, negative_slope=0.1), dim=-1), 'Mean Leaky ReLU with slope 0.1'),
        #     (lambda x: torch.mean(F.hardsigmoid(x), dim=-1), 'Mean HardSigmoid'),
        #     (lambda x: torch.mean(F.softplus(x), dim=-1), 'Mean Softplus'),
        # ]

    def compute_gate(self, x):
        gate_logits = self.gate(x)
        return gate_logits
    def topk_expert(self, gate_logits):
        """
        Selects the top-k experts based on the gating logits.

        This method computes the softmax of the gating logits to obtain the probabilities,
        then selects the top-k experts with the highest probabilities for each input sample.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            tuple:
                - weights (torch.Tensor): The softmax probabilities of the top-k experts.
                - selected_experts (torch.Tensor): Indices of the top-k experts.
                - gate_softmax (torch.Tensor): The softmax probabilities for all experts.
        """
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        return weights, selected_experts, gate_softmax
    def compute_singer_expert(self, x, w1, w2 = None):
        out = F.linear(x, w1)
      
        if w2 is not None:
            out = self.activation(out)
            out = F.linear(out, w2)
        return out
    
    def competition_policy_mlp_fasterx(self, x, compute_aff = lambda x: torch.mean(F.softplus(x), dim=-1)):
        """
        Implements the competition policy for expert selection.

        Args:
            x (tensor): Input tensor of shape (B, N, D), where:
                - B: Batch size
                - N: Sequence length
                - D: Input feature dimension

        Returns:
            weights (tensor): Tensor of shape (B, N, num_selected) representing the normalized weights for the selected experts.
            selected_experts (tensor): Tensor of shape (B, N, num_selected) containing the indices of the selected experts.
            affinity_softmax (tensor): Softmax probabilities of the affinity scores, with shape (B, N, num_of_experts).
        """
        B, N, D = x.shape
        expert_outputs = []
        # self.num_of_experts = 2
        breakpoint()
        x = x.unsqueeze(2).expand(B, N, self.num_of_experts, D).reshape(B, N * self.num_of_experts, D)
        indices = torch.arange(self.num_of_experts, device=x.device).unsqueeze(0)
        indices_expanded = indices.unsqueeze(0).expand(B, N, self.num_of_experts)
        indices_expanded = indices_expanded.unsqueeze(-1)
        selected_experts = indices_expanded.reshape(B, N * self.num_of_experts, 1) 
        weights_tmp = torch.ones(selected_experts.shape)
        # breakpoint()
        sel_indices = cvmm_prepare_sel2(selected_experts.int())

        scores = self.compute_scores(x, sel_indices) # nan
        
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = weights_tmp
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None
        out = cvmm(scores, sel_indices, self.values)
        expert_outputs = out.view(B, N, self.num_of_experts, D)
        
        affinity_scores = torch.mean(F.softplus(expert_outputs), dim = -1)
   
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(affinity_scores, 1)
        
        
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
      
        return weights, selected_experts, affinity_softmax, affinity_scores, None
    def competition_policy_mlp_faster(self, x):
        """
        Implements the competition policy for expert selection.

        Args:
            x (tensor): Input tensor of shape (B, N, D), where:
                - B: Batch size
                - N: Sequence length
                - D: Input feature dimension

        Returns:
            weights (tensor): Tensor of shape (B, N, num_selected) representing the normalized weights for the selected experts.
            selected_experts (tensor): Tensor of shape (B, N, num_selected) containing the indices of the selected experts.
            affinity_softmax (tensor): Softmax probabilities of the affinity scores, with shape (B, N, num_of_experts).
        """
        B, N, D = x.shape
        expert_outputs = []
        affinity_scores = torch.zeros(B, N, self.num_of_experts, device=x.device, dtype=x.dtype)
        expert_outputs = torch.matmul(x.view(-1, x.size(-1)), self.keys)
        expert_outputs = self.activation(expert_outputs)
        expert_outputs = torch.matmul(expert_outputs, self.values)
        expert_outputs = expert_outputs.transpose(1, 0) # (B*N, E, D)
        affinity_scores = torch.mean(F.softplus(expert_outputs), dim = -1)
        
        affinity_scores = affinity_scores.view(x.shape[0], x.shape[1], affinity_scores.shape[-1]) # (B, N, E)
        affinity_softmax = F.softmax(affinity_scores, dim=-1)
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        # compute input for diversity loss
        idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, self.num_selected, expert_outputs.size(-1))
        expert_outputs = expert_outputs.view(*x.shape[:2], *expert_outputs.shape[1:])
        topk_expert_outputs = torch.gather(expert_outputs, dim=2, index=idx_expanded)
        
        return weights, selected_experts, affinity_softmax, affinity_scores, topk_expert_outputs
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        in1 = in2 = x
        gate_logits = self.compute_gate(x)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        # weights, selected_experts, gate_softmax, gate_logits, topk_expert_outputs = self.competition_policy_mlp_faster(x)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
    
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
