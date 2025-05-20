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

@register_moe("competesmoe")
class CompeteSMoE(MoE):
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

        super().__init__(
                         dmodel=dmodel, 
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

        self.warm_up = args.warm_up  # Warm up expert with SMoE
        self.rate_flip = args.rate_flip
        self.total_steps = None
        self.current_steps = 0
        self.step_warm = None
        self.is_prob_flips = True
        # self.total_experts_out = None
        self.total_router_gate = []
        self.total_router_affinity = []
        self.total_diver = []
        self.total_steps = args.stop_after
        assert args.stop_after > 0, f"Warning: stop_after {args.stop_after} < 1, You must setting stop_after > 0"
        # self.set_total_steps(step=args.stop_after)
        self.prob_flips_final = {}
        self.max_compete_in_iter = args.max_compete_in_iter
        self.nb_diver = 0
        
        if getattr(self.args, "is_cosine", False):
            print("Active Consine Method")
        if getattr(self.args, "hybrid", False):
            print("Active Hybrid Method")
        if getattr(self.args, "is_norm_weight", False):
            print("Active Norm Weight Method")
        if getattr(self.args, "norm_sigmoid", False):
            print("Active Norm sigmoid Method")
        
    def set_total_steps(self, id_layer=0):
        """
        Initializes and configures the training steps for the current layer in the CompeteSMoE model.
        
        This implementation follows the architecture design from DeepSeek-V3 (https://github.com/deepseek-ai/DeepSeek-V3),
        adapting their Mixture-of-Experts (MoE) approach with our own competition-based expert selection mechanism.
        
        This method performs several key functions:
        1. Calculates warm-up and competition phases based on total training steps
        2. Creates a balanced candidate tensor for expert selection
        3. Ensures distributed synchronization across processes
        4. Manages the competition frequency to prevent exceeding maximum allowed competitions
        
        The method implements a sophisticated balancing mechanism that:
        - Tracks cumulative competition frequency across layers
        - Adjusts candidate positions when frequency thresholds are exceeded
        - Maintains distributed consistency across training processes
        
        Args:
            id_layer (int, optional): The identifier for the current layer. Defaults to 0.
            
        Returns:
            dict: A dictionary containing the updated probability flips for all layers,
                  where each key is a layer ID and each value is a boolean tensor
                  indicating competition opportunities.
                  
        Raises:
            ValueError: If the calculated flip steps are non-positive or if the
                       competition ratio becomes invalid.
        """
        # if self.training == False: return
        # Compute warm-up steps and determine the number of flip steps.
        self.step_warm = int(self.warm_up * self.total_steps)
        flip_steps = self.total_steps - self.step_warm
        self.flip_steps = flip_steps

        if flip_steps <= 0:
            raise ValueError("self.total_steps - self.step_warm must be greater than 0.")

        # Determine distributed rank and world size.
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Set up the device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def create_balanced_flip_current(cum_frequency):
            """
            Creates a boolean tensor for the current layer with shape [flip_steps].
            For each candidate position, if the random probability (based on self.rate_flip)
            is met but the cumulative frequency (from previous layers plus the current layer)
            would exceed self.max_compete_in_iter, the candidate is shifted left or right
            to find a valid position.

            Args:
                cum_frequency (Tensor): A tensor of shape [flip_steps] containing the cumulative
                                        count of True values from previous layers.

            Returns:
                Tensor: A boolean tensor indicating candidate flips for the current layer.
            """
            candidate_current = [False] * flip_steps  # Initialize candidates.
            candidate_origin = [False] * flip_steps
            freq_updated = cum_frequency.clone()        # Copy cumulative frequency for updates.

            for i in range(flip_steps):
                if torch.rand(1, device=device).item() < self.rate_flip:
                    candidate_origin[i] = True
                    if freq_updated[i] < self.max_compete_in_iter:
                    
                        candidate_current[i] = True
                        freq_updated[i] += 1
                    else:
                        found = False
                        # Try shifting to the left.
                        for j in range(i - 1, -1, -1):
                            if (freq_updated[j] < self.max_compete_in_iter) and (not candidate_current[j]):
                                candidate_current[j] = True
                                freq_updated[j] += 1
                                found = True
                                break
                        # If left shift fails, try shifting to the right.
                        if not found:
                            for j in range(i + 1, flip_steps):
                                if (freq_updated[j] < self.max_compete_in_iter) and (not candidate_current[j]):
                                    candidate_current[j] = True
                                    freq_updated[j] += 1
                                    found = True
                                    break
            # print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            # print(f"Layer {id_layer}: {(torch.tensor(candidate_current, dtype=torch.bool, device=device) != torch.tensor(candidate_origin, dtype=torch.bool, device=device)).sum()}")
            with open("./file_path.txt", "a") as f:
                # Write the new log entry at the top
                f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                f.write(f"Layer {id_layer}: {(torch.tensor(candidate_current, dtype=torch.bool, device=device) != torch.tensor(candidate_origin, dtype=torch.bool, device=device)).sum()}\n")
    
            return torch.tensor(candidate_current, dtype=torch.bool, device=device)

        # Only rank 0 creates the candidate tensor.
        if rank == 0:
            from tqdm import tqdm  # Optional progress display.
            import os

            # Compute cumulative frequency from previous layers.
            if self.prob_flips_final:
                frequency_on_compete = torch.zeros(flip_steps, dtype=torch.int, device=device)
                for _, v in self.prob_flips_final.items():
                    frequency_on_compete += v.int()
            else:
                frequency_on_compete = torch.zeros(flip_steps, dtype=torch.int, device=device)
                os.environ["start_max"] = '1'

            probs_current = create_balanced_flip_current(frequency_on_compete)
        else:
            # Other ranks create an empty tensor to receive the broadcast.
            probs_current = torch.empty(flip_steps, dtype=torch.bool, device=device)

        # Broadcast the candidate tensor to all processes if in distributed mode.
        if world_size > 1:
            dist.broadcast(probs_current, src=0)

        # Validate the candidate flips.
        count_true = probs_current.sum().item()
        count_false = flip_steps - count_true
        ratio_true = count_true / flip_steps
        ratio_false = count_false / flip_steps

        # if ratio_true == 0.0 or ratio_false == 0.0:
        #     raise ValueError("Invalid ratio of True or False in candidate flips.")

        # Assign the final candidate tensor for the current layer only once.
        self.prob_flips_final[id_layer] = probs_current
        
        # save file 
        import json
        save_weights = {}
        for layer in self.prob_flips_final.keys():
            save_weights[layer] = self.prob_flips_final[layer].tolist()
     
        if rank == 0:
            print(f"Updated prob_flips_final keys: {list(self.prob_flips_final.keys())}")
            print(f"\nCompute Competition Rate (Layer {id_layer}): {ratio_true}")
            print(f"Compute Router Policy Rate: {ratio_false}")
            print(f"Warm-up Steps: {self.step_warm}\n")

        self.is_prob_flips = False
        return self.prob_flips_final

    def update_aux_statistics(self, 
                              gate_logits = None, 
                              gate_softmax= None, 
                              selected_experts= None, 
                              experts_out= None, 
                              router_affinity_softmax= None, 
                              router_gate_softmax= None, 
                              is_competition = False
                            ):
        '''
            Update variable to compute loss. Because MoEUT using architechture share parameter. So
            only compute once for main parameters        
        
        '''
        if is_competition == False:
           
            self.total_selections.append(selected_experts)
            self.total_gate_logits.append(gate_logits)
            self.total_gate_softmax.append(gate_softmax)
          
        else:
            self.total_router_gate.append( router_gate_softmax)
            self.total_router_affinity.append( router_affinity_softmax)
    def pre_train_forward(self):
        ''' Reset all variable compute loss'''
        self.total_selections = []
        self.total_gate_softmax = []
        self.total_gate_logits = []
        # self.total_experts_out = []
        self.total_router_gate = []
        self.total_router_affinity = []
        # torch.cuda.empty_cache()
    def add_perplexity_reg(self):
        is_comp = len(self.total_router_affinity) !=0
        
        
        if len(self.total_gate_logits) !=0 :
            self.total_gate_logits = torch.stack(self.total_gate_logits, 1).flatten(1, 2)
            balance_loss = self.entropy_balance(self.total_gate_logits) * self.args.balance_loss_coef
            self.add_reg(lambda: balance_loss, f"{self.name_moe}_ebalance")
        if is_comp:
            self.total_router_affinity = torch.stack(self.total_router_affinity, 1).flatten(1, 2)
            self.total_router_gate = torch.stack(self.total_router_gate, 1).flatten(1, 2)
            router_loss = self.router_loss(affinity_softmax=self.total_router_affinity.detach(), gate_softmax=self.total_router_gate) * self.args.router_loss_coef
            # comp_balance_loss = self.entropy_balance(self.total_router_affinity) * self.args.balance_loss_coef/2
            # comp_diver_loss = torch.sum(torch.stack(self.total_diver)) / self.nb_diver

            self.add_reg(lambda: router_loss,  f"{self.name_moe}_router_loss")
            # self.add_reg(lambda: comp_balance_loss, f"{self.name_moe}_comp_ebalance")
            # self.add_reg(lambda: comp_diver_loss * self.args.balance_loss_coef / 2,  f"{self.name_moe}_comp_diver_loss")
            # self.add_reg(lambda: comp_diver_loss, "comp_diver_loss")
            self.nb_diver = 0
        self.pre_train_forward()
    def set_current_steps(self, step):
        self.current_steps = step
    def experts_diversity_loss(self, expert_outputs):
        """
        Compute the diversity loss between expert outputs.

        Args:
            expert_outputs (tensor): Tensor of shape (B, N, K, D), where:
                - B: Batch size
                - N: Sequence length
                - K: Number of selected experts
                - D: Dimension of each expert output

        Returns:
            loss (tensor): Scalar tensor representing the mean similarity between expert outputs.
        """
        if len(expert_outputs.shape) == 5:
            expert_outputs = expert_outputs.view(expert_outputs.shape[0], expert_outputs.shape[1] * expert_outputs.shape[2], *expert_outputs.shape[3:])
        B, N, K, D = expert_outputs.shape

        # Step 1: Normalize (L2-normalize) along the D dimension to calculate Cosine Similarity
        # Shape after normalization remains [B, N, K, D]
        normalized = F.normalize(expert_outputs, p=2, dim=-1)

        # Step 2: Reshape to a single batch for easier matrix multiplication
        # Reshape to [B*N, K, D]
        normalized_reshape = normalized.view(B*N, K, D)  # => [B*N, K, D]

        # Step 3: Calculate similarity matrix using matrix multiplication
        # [B*N, K, D] x [B*N, D, K] -> [B*N, K, K]
        similarity_matrix = torch.bmm(
            normalized_reshape, 
            normalized_reshape.transpose(1, 2)
        )  # => [B*N, K, K]

        # Step 4: Remove self-similarity (diagonal)
        # identity = [K, K], shape broadcast to [B*N, K, K]
        mask = 1 - torch.eye(K, device=expert_outputs.device)
        similarity_matrix = similarity_matrix * mask
        nb_diver = (similarity_matrix != 0).sum() 
        # Step 5: Calculate the mean of all batch, token, and expert pairs
        # similarity_matrix has shape [B*N, K, K]. Valid elements = B*N * K * (K-1)
        loss = similarity_matrix.mean()
        self.nb_diver += nb_diver
        return loss
    def compute_singer_expert(self, x, w1, w2 = None):
        out = F.linear(x, w1)
      
        if w2 is not None:
            out = self.activation(out)
            out = F.linear(out, w2)
        return out
   
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
        affinity_softmax = F.softmax(affinity_scores, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(affinity_scores, self.num_selected)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        # compute input for diversity loss
        idx_expanded = selected_experts.unsqueeze(-1).expand(B, N, self.num_selected, expert_outputs.size(-1))
        expert_outputs = expert_outputs.view(*x.shape[:2], *expert_outputs.shape[1:])
        topk_expert_outputs = torch.gather(expert_outputs, dim=2, index=idx_expanded)
        
        return weights, selected_experts, affinity_softmax, affinity_scores, topk_expert_outputs
    def topk_expert_softmax(self, gate_logits):
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
        weights, selected_experts = torch.topk(gate_logits, self.num_selected)
        weights = F.softmax(weights , dim=-1, dtype=torch.float)
        return weights, selected_experts, gate_softmax
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
    def compute_gate(self, x):
        if getattr(self.args, "is_cosine", False) == True and getattr(self.args, "is_norm_weight", False) == False:
            x_norm = F.normalize(x, p=2.0, dim=-1)
            gate_logits = F.linear(x_norm, F.normalize(self.w_gate, p=2.0, dim=-1), bias=None)
        elif getattr(self.args, "is_norm_weight", False):
            gate_logits = F.linear(x, F.normalize(self.w_gate, p=2.0, dim=-1), bias=None)
        else:
            gate_logits = self.gate(x)
        return gate_logits
    def router_policy(self, x, is_normal_mode = False):
        """
        Implements the standard routing policy using gate logits.

        Args:
            x (tensor): Input tensor of shape (B, N, D).
            is_normal_mode: is active model not competition
        Returns:
            weights (tensor): Normalized weights of the selected experts.
            selected_experts (tensor): Indices of the selected experts.
            gate_softmax (tensor): Softmax probabilities of the gate logits.
        """
        assert not (self.args.is_cosine == True and getattr(self.args, "is_norm_weight", False) == True), "Can not active  both  Cosine and Norm Weigh. Just use one method - Cosine or Norm Weigh to Normalization"
    
        gate_logits = self.compute_gate(x)
        if getattr(self.args, "norm_sigmoid", False) == True:
            gate_softmax = F.softmax(gate_logits, dim = -1, dtype=torch.float32)
            weights, selected_experts = torch.topk(gate_logits, self.num_selected)
            weights = F.sigmoid(weights / getattr(self.args, "scale_weight", 1.0))
        else:
            # Select experts using top-k gating
            weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
            
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
            
        return weights, selected_experts, gate_softmax, gate_logits

    def router_loss(self, gate_softmax, affinity_softmax):
        
        """
        Computes the router loss, which encourages the gate's softmax probabilities to match the affinity scores.

        Args:
            gate_softmax (tensor): Softmax probabilities from the gate logits of shape (B, N, num_of_experts).
            affinity_softmax (tensor): Softmax probabilities of the affinity scores of shape (B, N, num_of_experts).

        Returns:
            loss (tensor): Scalar tensor representing the mean squared error (MSE) between the gate and affinity softmax probabilities.
        """

        loss = F.mse_loss(gate_softmax, affinity_softmax)
        return loss
    
   

    def compute_moe_main(self, x, selected_experts, weights):
        
        # weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        sel_indices = cvmm_prepare_sel2(selected_experts.int())
        scores = self.compute_scores(x, sel_indices)
        
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = weights
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None
        
        out = cvmm(scores, sel_indices, self.values)
        return out
 
    def forward(self, x, return_id_experts = False, return_full = True, *args, **kwargs):
        # compute output
        id_layer = kwargs['id_layer']
        assert id_layer is not None, "Layer Id must to not None"
        is_comp = x.requires_grad and self.current_steps >= self.step_warm and self.prob_flips_final[id_layer][self.current_steps - self.step_warm].item() == 1
        gate_weights, gate_selected_experts, gate_softmax, gate_logits = self.router_policy(x, is_normal_mode = (is_comp == False) or (x.requires_grad == False))
        if is_comp:
            # Use competition policy for expert selection
            affinity_weights, affinity_selected_experts, affinity_softmax, affinity_logits, expert_outputs = self.competition_policy_mlp_faster(x)
          
            # Perform MoE computation using competition-selected experts
            out = self.compute_moe_main(
                selected_experts=affinity_selected_experts,
                weights=affinity_weights,
                x=x,
            )
            comp_diver_loss = self.experts_diversity_loss(expert_outputs) 
            self.add_reg(lambda: comp_diver_loss * self.args.balance_loss_coef_comp / 2, self.name_moe + "_comp_diver_loss")
            if self.args.balance_affinity:
                # define that need to balance experts when we are compute
                balance_experts = self.entropy_balance(affinity_softmax) 
                self.add_reg(lambda: balance_experts * self.args.balance_loss_coef_comp / 2, f"{self.name_moe}_comp_ebalance")
            if self.args.in_topk:
                gate_softmax_topk = torch.gather(gate_softmax, dim=-1, index=affinity_selected_experts)
                affinity_softmax_topk = torch.gather(affinity_softmax, dim=-1, index=affinity_selected_experts)
                router_loss = self.router_loss(
                    affinity_softmax=affinity_softmax_topk.detach(), 
                    gate_softmax=gate_softmax_topk) 
            elif self.args.hybrid:
                # print("hybrid learning")
                gate_softmax_topk = torch.gather(gate_softmax, dim=-1, index=affinity_selected_experts)
                affinity_softmax_topk = torch.gather(affinity_softmax, dim=-1, index=affinity_selected_experts)
                
                router_loss = self.router_loss(
                    affinity_softmax=affinity_softmax.detach(), 
                    gate_softmax=gate_softmax
                    
                )  + self.router_loss(
                    affinity_softmax=affinity_softmax_topk.detach(), 
                    gate_softmax=gate_softmax_topk
                    
                ) * self.args.router_theta 
            elif self.args.tribrid:
                # print("hybrid learning")
                # hybrid learning
                gate_softmax_topk = torch.gather(gate_softmax, dim=-1, index=affinity_selected_experts)
                affinity_softmax_topk = torch.gather(affinity_softmax, dim=-1, index=affinity_selected_experts)
                
                gate_softmax_topk_gate = torch.gather(gate_softmax, dim=-1, index=gate_selected_experts)
                affinity_softmax_topk_gate = torch.gather(affinity_softmax, dim=-1, index=gate_selected_experts)
                router_loss = self.router_loss(
                    affinity_softmax=affinity_softmax.detach(), 
                    gate_softmax=gate_softmax
                    
                )  + self.router_loss(
                    affinity_softmax=affinity_softmax_topk.detach(), 
                    gate_softmax=gate_softmax_topk
                    
                ) * self.args.router_theta + self.router_loss(
                    affinity_softmax=affinity_softmax_topk_gate.detach(), 
                    gate_softmax=gate_softmax_topk_gate
                    
                ) * self.args.router_theta 
            else:
                router_loss = self.router_loss(
                    affinity_softmax=affinity_softmax.detach(), 
                    gate_softmax=gate_softmax
                )
            # print(router_loss * self.args.router_loss_coef)
            self.add_reg(lambda: router_loss * self.args.router_loss_coef,  f"{self.name_moe}_router_loss")
            
        else:        
            
            out = self.compute_moe_main(
                selected_experts=gate_selected_experts,
                weights=gate_weights,
                x=x,
            ) 

            name = f"{self.name_moe}_ebalance"
            bal_loss = self.entropy_balance(gate_logits)* (self.args.balance_loss_coef / self.div)
            self.add_reg(lambda: bal_loss, name)
        self.layer += 1
        # save selected experts for analyst
        if self.args.test_only:
            self.add_dist_experts(selection=gate_selected_experts)
            self.add_dist_weight(weight=gate_weights)
            self.add_dist_weight(weight=gate_softmax, is_all=True)
        self.was_training = self.training
        res = out.view(*x.shape[:-1], self.v_dim)
        if self.o_bias is not None:
            res = res + self.o_bias
        return res
