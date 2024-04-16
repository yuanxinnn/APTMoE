
from typing import List
import torch
import torch.nn as nn
import time
from torch import Tensor
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast, List, Dict, Callable
from torch.nn import Module, ModuleList
import torch.nn.functional as F


def one_hot(tensor: torch.Tensor, num_classes: int) -> Tensor:
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def random_gating(logits: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    gates = F.softmax(logits, dim=1, dtype=torch.float)

    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    assert num_tokens % num_experts == 0

    indices1_s = torch.argmax(gates, dim=1)   
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    idx = torch.randperm(mask1.shape[1])
    mask1 = mask1[:, idx].view(mask1.size())
    
    expert_selection = torch.sum(mask1, dim=0)
    expert_selection = [s.item() for s in expert_selection]

    return expert_selection


class RandomGate(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts

    def power_law_logit(self, dim: int, num_experts: int):
        random_matrix = torch.rand(dim, num_experts) 
        
        exponent = 3
        exponents = torch.pow(torch.arange(1, num_experts+1, dtype=torch.float), -exponent)
        power_law_distribution = exponents / torch.sum(exponents)
        
        logits = torch.zeros(dim, num_experts)
        for i in range(dim):
            sampled_indices = torch.multinomial(power_law_distribution, num_samples=num_experts, replacement=True)
            logits[i, sampled_indices] = torch.poisson(random_matrix[i, sampled_indices])
        
        return logits

    def forward(self, input: torch.Tensor):
        logits = self.power_law_logit(input.shape[0], self.num_experts)
        gating_result = random_gating(logits)
        return gating_result


class ExpertFeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout):
        super(ExpertFeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        output = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        return output


class MoELayer(Module):
    def __init__(self, gate: Module, experts: Union[Module, ModuleList]):
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        
        self.real_expert_selection = []

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        
        d_model = input[0].shape[2] 
        reshaped_input = input[0].reshape(-1, d_model)  
        expert_outputs = []              
        
        self.real_expert_selection = self.gate(reshaped_input)

        chunks = torch.split(reshaped_input, self.real_expert_selection, dim=0)
            
        for chunk, expert in zip(chunks, self.experts):
            if chunk.shape[0] != 0:
                expert_outputs += [expert(chunk)]
        
        expert_output = torch.cat(expert_outputs, dim=0)
        expert_output = expert_output.reshape(input[0].shape)

        return expert_output
   

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.0,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=False,
        num_local_experts=1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.gate = RandomGate(d_model, num_local_experts)
        
        experts = nn.ModuleList(
            [ExpertFeedForwardLayer(d_model, dim_feedforward, activation, dropout) for expert_id in range(num_local_experts)]
        )
        
        self.moe_layer = MoELayer(self.gate, experts)


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src          
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = self._ff_block(x + self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        return self.moe_layer(x)


class Profiler():
    def __init__(
        self,
        max_num_tokens,
        d_model,
        dim_feedforward,
        num_experts,
        batch_size,
        seq_len,
    ) -> None:
        super().__init__()
        self.comp_cpu = [None] * (max_num_tokens+1)
        self.load_expert = None
        self.load_MHA = None
        self.load_Gate_8_experts = None
        self.load_Gate_16_experts = None
        self.load_Gate_64_experts = None
        self.max_num_tokens = max_num_tokens
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        
        self.batch_size = batch_size
        self.seq_len = seq_len

    def run_expert_profile(self):

        total_num_tokens = self.max_num_tokens
        
        activation = nn.ReLU()
        dropout = 0.1
        
        expert = ExpertFeedForwardLayer(self.d_model, self.dim_feedforward, activation, dropout)
        MHA = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=32, dropout=dropout)
        Gate_8_experts = torch.nn.Linear(in_features=self.d_model, out_features=8)
        Gate_16_experts = torch.nn.Linear(in_features=self.d_model, out_features=16)
        Gate_64_experts = torch.nn.Linear(in_features=self.d_model, out_features=64)

        repeat_times = 20
        # ============= Profile expert load Time
        expert.cuda()
        expert.to('cpu')    
        
        torch.cuda.synchronize()
        start_time = time.time()

        expert.cuda()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        self.load_expert = elapsed_time
        print(' load_expert =  '+str(elapsed_time))  

        # ============= Profile MHA Load Time
        MHA.cuda()
        MHA.to('cpu')    
        
        torch.cuda.synchronize()
        start_time = time.time()

        MHA.cuda()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        self.load_MHA = elapsed_time
        print(' load_MHA =  '+str(elapsed_time))  

        # ============= Profile Gate_8_experts Load Time
        Gate_8_experts.cuda()
        Gate_8_experts.to('cpu')    
        
        torch.cuda.synchronize()
        start_time = time.time()

        Gate_8_experts.cuda()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        self.load_Gate_8_experts = elapsed_time
        print(' load_Gate_8_experts =  '+str(elapsed_time))  

        # ============= Profile Gate_16_experts Load Time
        Gate_16_experts.cuda()
        Gate_16_experts.to('cpu')    
        
        torch.cuda.synchronize()
        start_time = time.time()

        Gate_16_experts.cuda()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        self.load_Gate_16_experts = elapsed_time
        print(' load_Gate_16_experts =  '+str(elapsed_time))

        # ============= Profile Gate_64_experts Load Time
        Gate_64_experts.cuda()
        Gate_64_experts.to('cpu')    
        
        torch.cuda.synchronize()
        start_time = time.time()

        Gate_64_experts.cuda()

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        self.load_Gate_64_experts = elapsed_time
        print(' load_Gate_64_experts =  '+str(elapsed_time))
 
 
        # ============= Profile CPU Comp Time
        self.comp_cpu[0] = 0
        expert.to('cpu')
        for num_tokens in range(1, total_num_tokens+1):
            
            inputs = torch.randn(num_tokens, d_model)
            outputs = expert(inputs)
            
            torch.cuda.synchronize()
            start_time = time.time()

            for step in range(repeat_times):
                outputs = expert(inputs)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            self.comp_cpu[num_tokens] = elapsed_time/repeat_times
            if num_tokens % 50 == 0:
                print('num tokens = ' + str(num_tokens) + ' comp_cpu[num_tokens] =  '+str(elapsed_time/repeat_times))


    def run_layer_profile(self):
        layer = TransformerDecoderLayer(d_model=self.d_model, nhead=32, dim_feedforward=dim_feedforward, dropout=0.0, 
                                            num_local_experts=self.num_experts).cuda()

        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        input_tensor = torch.rand(self.batch_size, self.seq_len, self.d_model).to("cuda")
        output = layer(input_tensor)
        target = torch.rand(output.size()).to("cuda")
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        print(
            "cuda {:d}: Peak memory {:.2f}GB ; Persistent memory {:.2f}GB"
                .format(0,torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30 , torch.cuda.memory_stats(0)["allocated_bytes.all.current"] / 2**30)
        )
            


num_experts = 8
d_model = 1024
dim_feedforward = 4096

seq_len = 128
batch_size = 128
num_micro_batch = 4
total_num_tokens = batch_size // num_micro_batch * seq_len

total_num_tokens = 64

profiler = Profiler(total_num_tokens, d_model, dim_feedforward, num_experts, batch_size, seq_len)

profiler.run_expert_profile()
profiler.run_layer_profile()

print('\n total_num_tokens = ' + str(total_num_tokens))
print('\n profiler.comp_cpu')
print(profiler.comp_cpu)
print('\n profiler.load_Gate_8_experts')
print(profiler.load_Gate_8_experts)
print('\n profiler.load_Gate_16_experts')
print(profiler.load_Gate_16_experts)
print('\n profiler.load_Gate_64_experts')
print(profiler.load_Gate_64_experts)
print('\n profiler.load_MHA')
print(profiler.load_MHA)
print('\n profiler.load_expert')
print(profiler.load_expert)