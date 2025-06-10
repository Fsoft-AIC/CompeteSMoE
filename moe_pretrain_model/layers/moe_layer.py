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
class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int, topk = 2,
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
                 args = None
                 ):

        super().__init__()
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

        self.coocurence = None
        self.prev_sel_oh = None
        
        self.gate = nn.Linear(dmodel, n_experts, bias = False)
        self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(dmodel, expert_size, bias = True),
                nn.ReLU(inplace=True),
                nn.Linear(expert_size, dmodel)) for _ in range(n_experts)])
        
        self.args = args
        self.total_selections = None
        self.total_gate_softmax = None
        self.total_gate_logits = None
        self.training = False
        self.num_experts = n_experts
        self.num_selected = topk
        self.num_of_experts = n_experts
        self.pre_train_forward()
        self.init_gate_weights()
        self.init_expert_weights()
    def init_gate_weights(self):
        """
            Initialize the weights and bias of the gating layer.
            We are make sure that gating of the xmoe same init weight setting with other algorithms 
        """
    
        gate_generator = torch.Generator(device=self.gate.weight.device)
        gate_generator.manual_seed(42)
        """
        Initialize the weights and bias of the gating layer.
        """
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02, generator=gate_generator)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, 0.0)
        print("Initializing weights and bias of the gating layer succefull")
    def init_expert_weights(self):
        """
        Initialize the weights and bias for all experts in self.experts.
        """
        for expert in self.experts:  # Duyệt qua từng expert trong ModuleList
            for layer in expert:  # Mỗi expert là một nn.Sequential chứa nhiều layers
                if isinstance(layer, nn.Linear):  # Chỉ khởi tạo trọng số của nn.Linear
                    nn.init.normal_(layer.weight, mean=0.0, std=self.expert_size ** -0.5 * self.weight_scale)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        print("Initializing weights and bias of the experts layer succefull")
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
        self.total_selections = None
        self.total_gate_softmax = None
        self.total_gate_logits = None
    def update_aux_statistics(self, gate_logits, gate_softmax, selected_experts ):
        '''Update variable to compute loss. Because MoEUT using architechture share parameter. So
            only compute once for main parameters        
        
        '''
        if self.total_selections is None:
            self.total_selections = selected_experts
            self.total_gate_logits = gate_logits
            self.total_gate_softmax = gate_softmax
        else:
            # breakpoint()
            self.total_selections = torch.cat([self.total_selections,selected_experts], dim=0)
            self.total_gate_logits = torch.cat([self.total_gate_logits, gate_logits], dim=0)
            self.total_gate_softmax = torch.cat([self.total_gate_softmax, gate_softmax], dim=0)

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
        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')
        one_hot_gate_indices = nn.functional.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_of_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_of_experts ** 2)
        return balance_loss
    def topk(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.topk(k, dim=-1, sorted=False)

    def logsoftmax_of_history(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate calculating logsumexp over a bigger batch than the current one. Will have stale values, but that
        # should not matter much later in training.
        return F.log_softmax(x, dim=-1)
    
    def add_perplexity_reg(self):
        # breakpoint()
        balance_loss = self.balanceloss(gate_softmax=self.total_gate_softmax, selected_experts=self.total_selections)* self.args.balance_loss_coef
        zlosses = self.zloss(gate_logits=self.total_gate_logits, gate_softmax=self.total_gate_softmax) * self.args.router_z_loss_coef
        # aux_losses = lambda: balance_loss + zlosses
        self.add_reg(lambda: balance_loss, "balance_loss")
        self.add_reg(lambda: zlosses, "zlosses")
        self.pre_train_forward()


    def compute_scores(self, input: torch.Tensor, index: CVMMSel) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is not None:
            scores = cvmm(input, index, self.keys)

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
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)
        
        return weights, selected_experts, gate_softmax
    def compute_moe(self, selected_experts, weights, results, x, experts = None):
        """
        Compute the output by routing through the selected experts.

        Args:
            selected_experts (torch.Tensor): Indices of the selected experts.
            weights (torch.Tensor): Weights of the selected experts.
            results (torch.Tensor): Tensor to store the results.
            x (torch.Tensor): Input tensor to be processed by the experts.

        Returns:
            torch.Tensor: The computed output from the selected experts.
        """

        infor_experts = {}

        for i in range(len(experts)):
            batch_idx, token_idx, topk_idx = torch.where(selected_experts == i)
            infor_experts[i] = [batch_idx, token_idx, topk_idx]
        
        for i, expert in enumerate(experts):
            batch_idx, token_idx, topk_idx = infor_experts[i]
           
            out_exp = expert(x[batch_idx, token_idx])
            results[batch_idx, token_idx] += weights[batch_idx, token_idx, topk_idx].unsqueeze(0).T * out_exp

        return results

    def forward(self, x, return_id_experts = False, *args, **kwargs):
        # compute output
        gate_logits = self.gate(x)
        # gate_softmax  = F.softmax(gate_logits, dim=-1)
        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        
        output = torch.zeros(x.shape[0], x.shape[1], self.k_dim, device=x.device, dtype=x.dtype)
        output = self.compute_moe(selected_experts, weights, output, x, experts= self.experts)
        # compute loss
        if self.training:
            self.update_aux_statistics(
                gate_logits=gate_logits,
                gate_softmax=gate_softmax,
                selected_experts=selected_experts
                )

        return output

    
