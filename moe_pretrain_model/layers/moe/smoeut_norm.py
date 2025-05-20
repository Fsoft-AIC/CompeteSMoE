from .moe import MoE
from typing import Any, Dict, Tuple, List, Union, Optional
from .register import register_moe
import torch.nn.functional as F
import torch
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

@register_moe("smoe_sigmoid")
class SMoEUTNorm(MoE):
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
        # gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        gate_sigmoid = torch.sigmoid(gate_logits)
        weights, selected_experts = torch.topk(gate_sigmoid, self.num_selected)
        
        return weights, selected_experts, gate_sigmoid
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        # compute output
        bsz, seq_len, h = x.shape
        in1 = in2 = x
        # breakpoint()
        gate_logits = self.compute_gate(x)
        # gate_softmax  = F.softmax(gate_logits, dim=-1)
        weights, selected_experts, gate_sigmoid = self.topk_expert(gate_logits=gate_logits)
        
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
        name = f"{self.name_moe}_balance"
        bal_loss = self.entropy_balance(gate_logits)* (self.args.balance_loss_coef / self.div)
        self.add_reg(lambda: bal_loss, name)
        if self.args.test_only:
            self.add_dist_experts(selection=selected_experts)
            self.add_dist_weight(weight=weights)
            self.add_dist_weight(weight=gate_sigmoid.softmax(dim = -1), is_all=True)
        return res
    
    # def att_forward(self, x, n_experts, n_copies, return_full = True, *args, **kwargs):
        
    #     if self.selection_dropout > 0 and self.training:
    #         x = F.dropout(x, self.selection_dropout)

    #     sel = self.compute_gate(x)
    #     sel = sel.view(*sel.shape[:-1], n_copies, -1)
    #     with torch.no_grad():
    #         if self.expert_dropout > 0 and self.training:
    #             mask = torch.rand_like(sel) < self.expert_dropout
    #             sel2 = sel.masked_fill(mask, float('-inf'))
    #         else:
    #             sel2 = sel
    #         _, sel_index = sel2.topk(self.num_selected, dim=-1, sorted=False)
    #     sel_val = torch.gather(sel, -1, sel_index)
    
    #     sel_val = sel_val.sigmoid()
    #     sel_val = sel_val / torch.sum(sel_val, dim=-1, keepdim=True).to(x.dtype)
        
    #     sel_index_shifted = (torch.arange(n_copies, device=sel_index.device, dtype=sel_index.dtype) * n_experts).unsqueeze(-1) + sel_index
    #     sel_index_pp = cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1).int(), sel_val)
    #     # compute loss
    #     if self.training:
    #         self.update_aux_statistics(
    #                 gate_logits=sel,
    #                 gate_softmax=sel.softmax(-1),
    #                 selected_experts=sel_index
    #             )

    #     return Selection(sel, sel_val, sel_index, sel_index_pp)

    # def compute_moe(self, x: torch.Tensor, sel: Selection) -> torch.Tensor:
    #     return cvmm(x, sel.sel_index, self.experts)