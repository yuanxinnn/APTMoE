from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import random

gumbel_map: Dict[torch.device, Callable] = {}


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: torch.Tensor, num_classes: int) -> Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: torch.Tensor, layer_id: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # NOTE(msb) softmax requires FP32: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
    gates = F.softmax(logits, dim=1, dtype=torch.float)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = 2S/E
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1) 
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)
    
    # expert_counts = torch.sum(mask1 + mask2, dim=0) 
    # print('layer '+str(layer_id) + ' expert_counts = ' + str(expert_counts))

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    gates1_s = (gates * mask1).sum(dim=1)  # einsum("se,se->s")
    gates2_s = (gates * mask2).sum(dim=1)  # einsum("se,se->s")
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    
    # dispatch_mask [512, 8, 128]
    
    return l_aux.to(logits.dtype), combine_weights.to(logits.dtype), dispatch_mask, None


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.layer_id = layer_id

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        # input [512, 1024]   logits [512, 8]
        logits = self.wg(input)
        return top2gating(logits, self.layer_id)


def random_gating(logits: torch.Tensor, layer_id: int) -> Tuple[Tensor, Tensor, Tensor]:
    gates = F.softmax(logits, dim=1, dtype=torch.float)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    assert num_tokens % num_experts == 0

    indices1_s = torch.argmax(gates, dim=1)   
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    # randomize the expert popularity
    idx = torch.randperm(mask1.shape[1])
    mask1 = mask1[:, idx].view(mask1.size())
    
    expert_selection = torch.sum(mask1, dim=0)
    # print('layer '+str(layer_id) + ' expert_selection = ' + str(expert_selection))
    
    expert_selection = [s.item() for s in expert_selection]

    return expert_selection


def random_list(original_selection):
    selection = original_selection
    
    max_num = max(selection)  
    max_index = selection.index(max_num)  

    decrease_range = min(max_num // 2, 100)
    decrease_amount = random.randint(1, decrease_range) 

    increase_amount = decrease_amount // (len(selection) - 1)

    selection[max_index] -= decrease_amount
    for i in range(len(selection)):
        if i != max_index:
            selection[i] += increase_amount

    return selection

def generate_exponent(num_experts, gini):
    exponent = 3
    
    if num_experts == 8:
        if gini == 0.2:
            exponent = 0.5
        elif gini == 0.4:
            exponent = 2
        elif gini == 0.6:
            exponent == 3
        elif gini == 0.8:
            exponent = 4.5
            
    elif num_experts == 16:
        if gini == 0.1:
            exponent = 0.5
        elif gini == 0.2:
            exponent = 1.8
        elif gini == 0.3:
            exponent = 2.3
        elif gini == 0.4:
            exponent = 2.8
            
    elif num_experts == 64:
        if gini == 0.05:
            exponent = 1.4
        elif gini == 0.1:
            exponent = 2
        elif gini == 0.15:
            exponent = 2.4
        elif gini == 0.2:
            exponent = 4.5
    
    return exponent

class RandomGate(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        layer_id: int,
        gini: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.layer_id = layer_id
        self.gini = gini
        self.on_GPU = False


    def power_law_logit(self, dim: int, num_experts: int):
        # seed = random.randint(1, 1000)
        # torch.manual_seed(seed)
        random_matrix = torch.rand(dim, num_experts) 
        
        exponent = generate_exponent(num_experts, self.gini)
        
        exponents = torch.pow(torch.arange(1, num_experts+1, dtype=torch.float), -exponent)
        power_law_distribution = exponents / torch.sum(exponents)

        # re-assign random matrix with given power law distribution
        logits = torch.zeros(dim, num_experts)
        for i in range(dim):
            sampled_indices = torch.multinomial(power_law_distribution, num_samples=num_experts, replacement=True)
            logits[i, sampled_indices] = torch.poisson(random_matrix[i, sampled_indices])
        
        return logits

    def forward(self, input: torch.Tensor):
        logits = self.power_law_logit(input.shape[0], self.num_experts)
        gating_result = random_gating(logits, self.layer_id)
        # return random_list(gating_result)
        return gating_result



