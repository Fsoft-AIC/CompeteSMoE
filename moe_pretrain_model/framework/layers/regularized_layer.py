import torch
import torch.nn
from typing import Dict, Any, Callable, Tuple, Optional, Set, Union, Iterable


from typing import Callable, Dict
import torch

class RegularizedLayer:
    def __init__(self) -> None:
        super().__init__()
        # Stores the accumulated regularization loss for each type
        self.reg_accumulated = {}
        # Stores the count of times each regularization loss was added
        self.reg_counts_n = {}
        self.regularization_present = False

    @property
    def reg_enabled(self) -> bool:
        """
        Checks if regularization is enabled. 
        Regularization is only enabled during training and if regularization is present.
        """
        return self.training and self.regularization_present

    def add_reg(self, loss_fn: Callable[[], torch.Tensor], name: str = "reg"):
        """
        Adds a regularization loss to the accumulated losses.

        Args:
            loss_fn (Callable): A function that returns a tensor representing the regularization loss.
            name (str): The name of the regularization loss type (default: "reg").
        """
        if self.reg_enabled:
            reg_value = loss_fn()  # Compute loss at the time of the call
            
            # Accumulate loss and update the count
            if name in self.reg_accumulated:
                self.reg_accumulated[name] += reg_value
                self.reg_counts_n[name] += 1
            else:
                self.reg_accumulated[name] = reg_value
                self.reg_counts_n[name] = 1
        
    def get_reg_loss(self) -> Dict[str, torch.Tensor]:
        """
        Computes the mean regularization loss for each type.

        Returns:
            dict: A dictionary where keys are regularization loss names and values are mean loss tensors.
        """
        reg_loss = {
            name: self.reg_accumulated[name] / self.reg_counts_n[name]
            for name in self.reg_accumulated
        }
        # if len(reg_loss) > 0:
        #     print(self.reg_counts_n)
        # Reset accumulated losses and counts after retrieval
        self.reg_accumulated = {}
        self.reg_counts_n = {}

        return reg_loss


class LayerRegularizer:
    def __init__(self, module: Union[torch.nn.Module, Iterable[torch.nn.Module]], stop_after: Optional[int] = None, scales: Dict[str, float] = {},
                 lin_decay: Set[str] = set(), options: Dict[str, Any] = {}):

        self.modules = []
        self.scales = scales
        self.stop_after = stop_after
        self.lin_decay = set(lin_decay)

        if self.lin_decay and stop_after is None:
            raise ValueError("Please specify stop_after when using lin_decay.")

        if isinstance(module, torch.nn.Module):
            self.add_module(module)
        else:
            for m in module:
                self.add_module(m)

    def add_module(self, module: torch.nn.Module):
        for n, m in module.named_modules():
            if isinstance(m, RegularizedLayer):
                # print(n)
                self.modules.append((n, m))
                m.regularization_present = True
                # breakpoint()
        # breakpoint()
    def get(self, iter: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res = {}
        for _, m in self.modules:
            for k, v in m.get_reg_loss().items():
                res[k] = res.get(k, 0) + v
        to_log = {k: v.detach() for k, v in res.items()}
        for k, v in res.items():
           
            res[k] = v * self.scales.get(k, 1)
            
        for k in self.lin_decay:
            res[k] *= 1 - iter / self.stop_after
        # breakpoint()
        return sum(res.values()), to_log
