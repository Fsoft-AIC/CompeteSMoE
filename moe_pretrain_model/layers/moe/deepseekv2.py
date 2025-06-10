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

@register_moe("deepseekv2")
class DeepSeekV2(MoE):
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
        
        self.n_shared_experts = 1
        self.values_shared = torch.nn.Parameter(torch.empty(1, self.expert_size * self.n_shared_experts, self.v_dim))
        
        sel_count = self.n_experts
        real_size_shared = self.n_shared_experts * self.expert_size
        self.get_initializer()(self.values_shared, std=real_size_shared ** -0.5 * weight_scale)
        #### ================
        mid_layer_scale = weight_scale
        self.keys_shared = torch.nn.Parameter(torch.empty(1, self.k_vec_dim, self.expert_size * self.n_shared_experts))
        self.get_initializer()(self.keys_shared, std=dmodel ** -0.5 * mid_layer_scale)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.bias_shared = torch.nn.Parameter(torch.zeros(1, self.expert_size * self.n_shared_experts))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.bias_shared = None
            self.o_bias = None
    def compute_gate(self, x):
        gate_logits = self.gate(x)
        return gate_logits
    
    def compute_scores(self, input: torch.Tensor, index: CVMMSel, is_shared: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not is_shared:
            if self.keys is not None:
                scores = cvmm(input, index, self.keys)
            if self.bias is not None:
                scores = scores + self.bias[index.raw_sel]
        else:
            if self.keys_shared is not None:
                scores = cvmm(input, index, self.keys_shared)
            if self.bias_shared is not None:
                scores = scores + self.bias_shared[index.raw_sel]


        scores = self.activation(scores)

        return scores
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        in1 = in2 = x
        
        gate_logits = self.compute_gate(x)
        
        weights, selected_experts = torch.topk(gate_logits, self.num_selected)
        
        weights = F.softmax(weights, dim = -1).to(x.dtype)
        
        sel_indices = cvmm_prepare_sel2(selected_experts.int())
        scores = self.compute_scores(in2, sel_indices)
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = weights
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None
        
        out = cvmm(scores, sel_indices, self.values)
        ### output of shared experts
        sel_index_shared = torch.zeros(selected_experts.size(0), selected_experts.size(1), 1, device=selected_experts.device)
        sel_val_shared = torch.ones(selected_experts.size(0), selected_experts.size(1), 1, device=selected_experts.device)

        sel_indices_shared = cvmm_prepare_sel2(sel_index_shared.int())
        scores_shared = self.compute_scores(in1, sel_indices_shared, is_shared=True)

        sel_indices_shared = sel_indices_shared.clone()
        sel_indices_shared.reduction_weight = sel_val_shared
        sel_indices_shared.sel_index = sel_indices_shared.out_index
        sel_indices_shared.out_index = None

        out_shared = cvmm(scores_shared, sel_indices_shared, self.values_shared)
        out = out + out_shared
        # ----------
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
            self.add_dist_weight(weight=F.softmax(gate_logits), is_all=True)
        
        return res
    