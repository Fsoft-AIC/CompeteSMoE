import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from moe_model.model.moe.register import register_moe
import torch.nn.functional as F
import copy
import loguru

from .moe import MoeLayer


@register_moe("smoe_share")
class MoEShareLayer(MoeLayer):
    def __init__(self, in_embed_dim=768, out_embed_dim=768, num_of_experts=4, num_selected=2, expert=None, args=None):
        super().__init__()
        '''
        We are implementing the following with the Deepseek repository.
        '''
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.num_of_experts = num_of_experts
        self.num_selected = num_selected
        self.args = args

        # initialize the router and expert
        if expert is None:
            print("initialize the selected expert with random init")
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(self.in_embed_dim, self.out_embed_dim), 
                nn.GELU(), 
                nn.Linear(self.out_embed_dim, self.out_embed_dim)) for _ in range(self.num_of_experts)])
        else:
            print("Initialize the selected expert with deep copy expert")
            self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(self.num_of_experts)])

        self.num_selected, self.num_of_experts = self.num_selected - 1, self.num_of_experts - 1
        self.gate = nn.Linear(self.in_embed_dim, self.num_of_experts, bias=False)

        self.init_gate_weights()
    def forward(self, x, return_id_experts = False, is_vision = False):
        gate_logits = self.gate(x)

        output = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)

        weights, selected_experts, gate_softmax = self.topk_expert(gate_logits=gate_logits)
        
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x.dtype)
        
        output_selected = torch.zeros(x.shape[0], x.shape[1], self.out_embed_dim, device=x.device, dtype=x.dtype)
        
        output_selected = self.compute_moe(selected_experts, weights, output_selected, x)
        
        output_shared = self.experts[self.num_of_experts](x)
        # we are use sparse upcycling strategy to combine the output of shared experts and selected experts need convert to range value of the dense. 
        output+= output_shared*0.5 + output_selected*0.5
        auxiliary_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        infor_aux = {}
        if x.requires_grad: 
            # compute loss
            auxiliary_loss, balance_loss, router_z_loss = self.combine_loss(selected_experts, gate_softmax, gate_logits)
            infor_aux = {
                "balance_loss": balance_loss.clone().detach(),
                "router_z_loss": router_z_loss.clone().detach()
            }
      
        return output, auxiliary_loss, None, infor_aux
