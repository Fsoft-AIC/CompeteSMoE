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
import torch


# Selection = namedtuple('moe', ['output', 'weights', 'selected_experts', 'gate_softmax', 'gate_logits'])
Selection = namedtuple('Selection', ['raw_sel', 'sel_val', 'raw_sel_index', 'sel_index'])

class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, 
                 dmodel: int, 
                 n_experts: int, 
                 expert_size: int, 
                 n_heads: int, 
                 std_gate: float,
                 std_expert: float,
                 topk = 2,
                 dropout: float = 0, weight_scale: float = 1.0,
                 selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
                 perplexity_reg_mode: str="step",
                 activation_after_topk: bool = False,
                 activation = lambda x: F.relu(x, inplace=True),
                  sel_bias: bool = False,
                 bias: bool = False,
                  v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 sync_distributed: bool = False,
                 selection_dropout: float = 0.0,
                 log_interval: Optional[int] = 100,
                 args = None,
                 is_att = False,
                 
                 out_dmodel = None,
                 inp_expert = None, 
                 out_expert = None,
                 ):

        super().__init__()
        self.is_att = is_att
        self.iter = 0
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.perplexity_reg = perplexity_reg
        self.k_vec_dim = self.k_dim
        self.n_heads = n_heads
        self.perplexity_reg_mode = perplexity_reg_mode
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.weight_scale = weight_scale
        self.layer = 0
        self.initalized = False
        self.was_training = True
        self.expert_dropout = expert_dropout
        self.reg_counts = 0
        self.sync_distributed = sync_distributed and torch.distributed.is_initialized()
        self.record_all_expert_sel_counts = False
        self.selection_dropout = selection_dropout
        self.log_interval = log_interval
        self.out_dmodel = out_dmodel if out_dmodel is not None else dmodel
        self.coocurence = None
        self.prev_sel_oh = None
        
        # self.gate = nn.Linear(dmodel, n_experts, bias = False)
        
        self.div = 1
 
        self.name_moe = "mlp"
        self.args = args
        self.total_selections = []
        self.total_gate_softmax = []
        self.total_gate_logits = []
        self.training = False
        self.num_experts = n_experts
        self.num_selected = topk # warning with MLP we get number of expert is n_head
        self.num_of_experts = n_experts
        self.sel_weight_scale = weight_scale
        mid_layer_scale =  weight_scale
        real_size = self.size
        self.real_n_experts = 1
        if is_att:
            self.w_gate = torch.nn.Parameter(torch.randn(self.n_experts, dmodel) * std_gate)
            self.renorm_rows(self.w_gate)
            self.gate = lambda x: F.linear(x, self.w_gate, None)
            self.div = 10
            self.real_n_experts  = self.n_heads
            self.register_parameter('experts', torch.nn.Parameter(torch.randn(n_experts, inp_expert, out_expert) * std_expert))
            
        else:
            self.w_gate = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))
            self.gate = lambda x: F.linear(x, self.w_gate, None)
            self.get_initializer()(self.w_gate, std=self.k_vec_dim ** -0.5 * self.sel_weight_scale)
            
            self.register_parameter('values', torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim)))
            self.register_parameter('keys', torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size)))
            self.get_initializer()(self.keys, std=dmodel ** -0.5 * mid_layer_scale)
            self.get_initializer()(self.values, std=real_size ** -0.5 * weight_scale)
            self.num_selected = self.n_heads 
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.n_experts, self.expert_size))
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.o_bias = None
        self.pre_train_forward()
        self.dist_experts = None
        self.entropy_expert_selected = []
        self.entropy_expert_all = []
        
    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())
    def entropy(self, prob_dist):
        """
        Compute the entropy of a probability distribution.

        Args:
            prob_dist (torch.Tensor): A tensor of shape (batch_size, num_classes)
                                    representing probability distributions.

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing entropy values.
        """
        # Add a small epsilon to prevent log(0)
        epsilon = 1e-18  
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + epsilon), dim=-1)
        return entropy
    def get_dist_experts(self):
        return self.dist_experts
    
    def add_dist_experts(self, selection = None):
        assert selection is not None, "Selection must to not None"
        selection = selection.reshape(-1, selection.shape[-1])
        one_hot_selected = F.one_hot(selection, num_classes=self.num_of_experts // self.real_n_experts)
        one_hot_selected = one_hot_selected.reshape(-1, one_hot_selected.shape[-1]).sum(-2)
        if self.dist_experts is None:
            self.dist_experts = one_hot_selected
        else:
            self.dist_experts += one_hot_selected

    def add_dist_weight(self, weight, is_all = False):
        weight = self.entropy(weight).mean()
        if is_all:
            self.entropy_expert_all.append(weight)
        else:
            self.entropy_expert_selected.append(weight)
    def get_weight_dist(self):
        return {
            'entropy_all': torch.stack(self.entropy_expert_all).mean().item(),
            'entropy_topk': torch.stack(self.entropy_expert_selected).mean().item()
        }
    def balance_loss_standard(self, sel_aux: torch.Tensor, sel_index: torch.Tensor, bsz: int, seq_len: int):
        aux_loss = torch.zeros(bsz, self.n_experts, device=sel_aux.device)
        aux_loss.scatter_add_(1, sel_index.view(bsz, seq_len * self.n_heads), torch.ones(bsz, seq_len * self.n_heads, device=sel_aux.device)).div_(seq_len * self.n_heads / self.n_experts)

        return (aux_loss * sel_aux.mean(dim = 1)).sum(dim = 1).mean()

            
    def init_gate_weights(self, std = 0.02):
        """
            Initialize the weights and bias of the gating layer.
            We are make sure that gating of the xmoe same init weight setting with other algorithms 
        """
    
        # gate_generator = torch.Generator(device=self.gate.weight.device)
        # gate_generator.manual_seed(42)
        """
        Initialize the weights and bias of the gating layer.
        """
        nn.init.normal_(self.gate.weight, mean=0.0, std=std)
        if self.is_att :
            self.renorm_rows(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0.0)
        print("Initializing weights and bias of the gating layer succefull")
    def keys_to_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        k = keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
        return k.permute(0, 2, 1).contiguous().view(-1, self.k_vec_dim)

    def keys_from_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        return keys.view(self.n_experts, self.expert_size, self.k_vec_dim).permute(0, 2, 1).contiguous().view(self.n_experts * self.k_vec_dim, self.expert_size)


    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())


    def fix_expert_sel_init(self):
        with torch.no_grad():
            self.renorm_keep_std(self.expert_sel, dim=1)

    def get_initializer(self):
        return torch.nn.init.normal_

    def sparse_matmul(self, indices: torch.Tensor, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.embedding_bag(indices, weight.type_as(values), per_sample_weights=values, mode="sum", sparse=False)

    def ani(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        chunk_size = 32

        xnorm = F.normalize(x, 2, dim=-1)

        accu = 0
        for i in range(0, x.shape[0], chunk_size):
            a = xnorm[i: i + chunk_size]
            sims = xnorm @ a.T
            sims[i : i + chunk_size].fill_diagonal_(0)
            accu += sims.sum()

        return accu / (x.shape[0] * (x.shape[0] - 1))

    def log_expert_sel_usage(self, prefix: str, channel_sel_counts: torch.Tensor):
        sel_nonzero = (channel_sel_counts != 0).type(torch.float).sum(axis=-1) / self.expert_size
        self.log(f"{prefix}/mean", sel_nonzero.mean())
        self.log(f"{prefix}/min", sel_nonzero.min())
        self.log(f"{prefix}/max", sel_nonzero.max())

    def pre_train_forward(self):
        ''' Reset all variable compute loss'''
        self.total_selections = []
        self.total_gate_softmax = []
        self.total_gate_logits = []
    def update_aux_statistics(self, gate_logits, gate_softmax, selected_experts ):
        '''Update variable to compute loss. Because MoEUT using architechture share parameter. So
            only compute once for main parameters        
        
        '''
  
        self.total_selections.append(selected_experts)
        self.total_gate_logits.append(gate_logits)
        self.total_gate_softmax.append(gate_softmax)

    def before_loss(self):
        self.add_perplexity_reg()
        if self.training:
            self.iter += 1
    def zloss(self, gate_logits, gate_softmax = None):
        """
        Computes the z-loss based on the gating logits.

        The z-loss is a measure of how uniformly the gating logits are distributed.
        It encourages sparsity in the gating distribution by penalizing the logarithm
        of the sum of the exponentials of the logits.

        Args:
            gate_logits (torch.Tensor): The logits from the gating network.

        Returns:
            torch.Tensor: The computed z-loss value.
        """
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)
        router_z_loss = router_z_loss.mean()
        return router_z_loss

    def balanceloss(self, selected_experts, gate_softmax):
        """
        Computes the balance loss for the selected experts.

        This loss measures how evenly the gating softmax distribution is distributed
        among the selected experts. It encourages a balanced distribution across experts
        by comparing the density of selected experts with the density of the overall gating softmax.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts
            gate_softmax (torch.Tensor): Softmax probabilities for each expert

        Returns:
            torch.Tensor: The computed balance loss value.
        """
        if self.is_att:
            # Because attention get topk expert for each head in a token. 
            # So I need to balance on head with minimize is 1 / (N / head) 
            # with N is total experts
            b, n, h, k = gate_softmax.shape
            density_1_proxy = reduce(gate_softmax, 'b n h k -> b h k', 'mean') 
            selected_experts_one_hot = nn.functional.one_hot(selected_experts, num_classes=k).float()[:, :, :, 0, :]
            density_1 = reduce(selected_experts_one_hot, 'b t h n -> b h n', 'mean')
            balance_loss = (density_1_proxy * density_1).mean() * float(k ** 2)
        else:
            density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
            one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts // self.real_n_experts).float()[0]
            density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
            balance_loss = (density_1_proxy * density_1).mean() * float(self.num_of_experts ** 2)
            
        return balance_loss
    def entropy_balance(self, sel):
        d = -2 
        if self.is_att:
            d = -3 # attention has head dim
        else:
            sel = sel.flatten(1, -2)
        sel_d = self.logsoftmax_of_history(sel)
        sel_d = framework.utils.distributed_ops.log_mean(sel_d, d, False)
        loss =  - utils.entropy_l(sel_d).mean()
        return loss
    def topk(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.topk(k, dim=-1, sorted=False)

    def logsoftmax_of_history(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate calculating logsumexp over a bigger batch than the current one. Will have stale values, but that
        # should not matter much later in training.
        return F.log_softmax(x, dim=-1)
    
    def add_perplexity_reg(self):
        
        # self.total_gate_logits = torch.stack(self.total_gate_logits, 1).flatten(1, 2)
        # self.total_gate_softmax = torch.stack(self.total_gate_softmax, 1).flatten(1, 2)
        # self.total_selections = torch.stack(self.total_selections, 1).flatten(1, 2)

        # bal_loss = self.entropy_balance(self.total_gate_logits)* (self.args.balance_loss_coef / self.div)
        
        # name = f"{self.name_moe}_ebalance"
        # self.add_reg(lambda: bal_loss, name)
        # balance_loss = self.balanceloss(gate_softmax=self.total_gate_softmax, selected_experts=self.total_selections)* (self.args.balance_loss_coef / self.div)
        
        # zlosses = self.zloss(gate_logits=self.total_gate_logits, gate_softmax=self.total_gate_softmax) * (self.args.router_z_loss_coef / self.div)
        
        # # aux_losses = lambda: balance_loss + zlosses
        # self.add_reg(lambda: balance_loss, "balance_loss")
        # self.add_reg(lambda: zlosses, "zlosses")
        self.pre_train_forward()
    

    def sel_activation(self, sel: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_sel = sel
        if self.selection_mode in {"sigmoid"}:
            sel = torch.sigmoid(sel)
        elif self.selection_mode in {"gate"}:
            sel = F.softmax(sel, dim=-1)
            with torch.no_grad():
                self.log("expert_rel_perplexity_per_selection", utils.relative_perplexity(sel).mean())
        else:
            assert False

        return sel, reg_sel
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
        # gate_sigmoid = torch.sigmoid(gate_logits)
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        
        return weights, selected_experts, gate_softmax
   
    def compute_gate(self, x):
        return self.gate(x)
    def compute_scores(self, input: torch.Tensor, index: CVMMSel) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.keys is not None:
            scores = cvmm(input, index, self.keys) # torch.isnan(self.keys).any()

        if self.bias is not None:
            scores = scores + self.bias[index.raw_sel]

        scores = self.activation(scores)

        plot_training = self.train and self.log_interval is not None and self.iter % self.log_interval == 0
        if plot_training:
            with torch.no_grad():
                gt0 = (scores > 0).float()
                gt0_s = gt0.sum()

                if plot_training:
                    self.log("relu_pass_rate", gt0_s / scores.numel())

        return scores
    # def compute_gate(self, x)
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        # compute output
        in1 = in2 = x
        # breakpoint()
        gate_logits = self.compute_gate(x)
        # gate_softmax  = F.softmax(gate_logits, dim=-1)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        if self.training is False:
            self.add_dist_experts(selection=selected_experts)
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
        # compute loss
        # if self.training:
        #     self.update_aux_statistics(
        #             gate_logits=gate_logits,
        #             gate_softmax=gate_softmax,
        #             selected_experts=selected_experts
        #         )
        
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
    #     if self.training is False:
    #         self.add_dist_experts(selection=sel_index)
    #     sel_val = sel_val.sigmoid()
        
    #     sel_index_shifted = (torch.arange(n_copies, device=sel_index.device, dtype=sel_index.dtype) * n_experts).unsqueeze(-1) + sel_index
    #     sel_index_pp = cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1).int(), sel_val)
    #     # compute loss
    #     if self.training:
    #         self.update_aux_statistics(
    #                 gate_logits=sel,
    #                 gate_softmax=sel.softmax(-1),
    #                 selected_experts=sel_index
    #             )
    #     #Selection = namedtuple('Selection', ['raw_sel', 'sel_val', 'raw_sel_index', 'sel_index'])

    #     return Selection(sel, sel_val, sel_index, sel_index_pp)

    # def compute_moe(self, x: torch.Tensor, sel: Selection) -> torch.Tensor:
    #     return cvmm(x, sel.sel_index, self.experts)
