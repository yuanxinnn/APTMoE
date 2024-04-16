import math
import torch
import torch.nn as nn
from .moe_layer import MoELayer
from .top2gate import Top2Gate, RandomGate
from typing import Any
import copy


class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp_sqrt = math.sqrt(ninp)

        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.ninp_sqrt

class PositionalEncodingLayer(nn.Module):
    """PositionalEncoding layer for a given Transformer model."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class OffloadInputBegin(torch.autograd.Function):
    @staticmethod
    # move input to CPU before forward computation
    def forward(ctx, inputs, gpu_device, layer_id, expert_id):
        ctx.device = inputs.device
        inputs = inputs.to(device = 'cpu')
        ctx.layer_id = layer_id
        ctx.gpu_device = gpu_device
        return inputs

    @staticmethod
    # move gradient to GPU after backward computation
    def backward(ctx, grad_output):
        grad_output = grad_output.to(device = ctx.gpu_device)
        return grad_output, None, None, None, None

class OffloadInputEnd(torch.autograd.Function):
    @staticmethod
    # move input to GPU after forward computation
    def forward(ctx, inputs, gpu_device, layer_id, expert_id):
        ctx.device = inputs.device
        inputs = inputs.to(device = gpu_device)
        ctx.layer_id = layer_id
        ctx.expert_id = expert_id
        return inputs

    @staticmethod
    # move gradient to CPU before backward computation
    # backward computation is executed on CPU
    def backward(ctx, grad_output):
        grad_output = grad_output.to(device = 'cpu')
        return grad_output, None, None, None, None


class InnerFeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""
    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super(InnerFeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        output = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        return output


class Test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, model, layer_id, expert_id):
        ctx.layer_id = layer_id
        ctx.expert_id = expert_id
        ctx.model = model
        # print('fwd comp layer '+str(ctx.layer_id) + ' hot expert_id = '+str(ctx.expert_id))
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        param_device = next(ctx.model.parameters()).device
        if grad_output.device != param_device:
            print('bwd comp layer '+str(ctx.layer_id) + ' expert_id = '+str(ctx.expert_id) +' grad_output.device = '+str(grad_output.device)+ ' param_device = '+str(param_device))
        for name, param in ctx.model.named_parameters():
            if param.grad is not None:
                if param.device != param.grad.device:
                    print(f'Layer: {name} - Param: {param.device} - Gradient: {param.grad.device}')
        return grad_output, None, None, None, None


class ExpertFeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""
    def __init__(self, d_model, dim_feedforward, activation, dropout, pipeline='GPipe', stage_id=0, layer_id=0, expert_id=-1) -> None:
        super(ExpertFeedForwardLayer, self).__init__()
        self.inner_model = InnerFeedForwardLayer(d_model, dim_feedforward, activation, dropout)

        self.on_GPU = True if pipeline == 'GPipe' else False
        self.stage_id = stage_id
        self.layer_id = layer_id
        self.expert_id = expert_id
        

    def forward(self, x):
        # if the expert is executed on CPU, then we need to move input to CPU
        if self.on_GPU == False:
            gpu_device = x.device
            x = OffloadInputBegin.apply(x, gpu_device, self.layer_id, self.expert_id)
            x = self.inner_model(x)
            x = OffloadInputEnd.apply(x, gpu_device, self.layer_id, self.expert_id)
        else:
            x = self.inner_model(x)
        # x = Test.apply(x, self, self.layer_id, self.expert_id)
        return x


class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout, layer_id) -> None:
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.on_GPU = False
        self.layer_id = layer_id

    def forward(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=False,
        is_moe=False,
        num_local_experts=1,
        stage_id=0, 
        layer_id=0,
        ndecoder=0,
        pipeline = 'GPipe',
        comp_stream = None, 
        gini = None, 
        comm_scheduler = None,
        R_solver = None,
        inter_stage_only = None
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout, layer_id=layer_id)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.stage_id = stage_id
        self.layer_id = layer_id
        self.ndecoder = ndecoder
        self.comp_stream = comp_stream
        self.CommScheduler = comm_scheduler
        self.gini = gini
        self.inter_stage_only = inter_stage_only

        self.MHA_event = torch.cuda.Event()
        self.Gate_event = torch.cuda.Event()
        self.expert_events = [torch.cuda.Event() for _ in range(num_local_experts)]
    
        self.is_moe = is_moe
        if is_moe:
            self.gate = RandomGate(d_model, num_local_experts, self.layer_id, self.gini)
            self.next_gate = RandomGate(d_model, num_local_experts, self.layer_id, self.gini)
            # self.gate = Top2Gate(d_model, num_local_experts, self.layer_id)
            # self.next_gate = Top2Gate(d_model, num_local_experts, self.layer_id)
            
            experts = nn.ModuleList(
                [ExpertFeedForwardLayer(d_model, dim_feedforward, activation, dropout, pipeline, stage_id, layer_id, expert_id) for expert_id in range(num_local_experts)]
            )
            
            self.moe_layer = MoELayer(self.gate, self.next_gate, self.layer_id, self.ndecoder, experts, pipeline, self.comp_stream, 
                                      self.CommScheduler, R_solver, self.Gate_event, self.expert_events, self.inter_stage_only)
        else:
            self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

            
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
        while self.self_attn.on_GPU == False:
            self.CommScheduler.load_execute_with_priority()
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout(x)

    def _ff_block(self, x):
        if self.is_moe:
            return self.moe_layer(x)
        else:
            return self.ff_block(x)



class TransformerLM(nn.Sequential):
    """A GPT-2 based nn.Sequential language model."""

    def __init__(self, ninp, nhead, nhid, dropout, ndecoder, is_moe=True, num_local_experts=1, use_fp16=False, 
                 pipeline = 'GPipe', stage_id=-1, gini=None, comm_scheduler=None, R_solver=None, inter_stage_only=None):
        self.comp_stream = torch.cuda.Stream()
        self.comm_scheduler = comm_scheduler
        self.R_solver = R_solver
        self.stage_id = stage_id
        
        layers = []
        
        for layer_id in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout, is_moe=is_moe, num_local_experts=num_local_experts, stage_id=stage_id, layer_id=layer_id+(stage_id*ndecoder),
                            ndecoder=ndecoder, pipeline = pipeline, comp_stream=self.comp_stream, gini=gini, comm_scheduler=self.comm_scheduler, R_solver=self.R_solver, inter_stage_only=inter_stage_only))
        
        super(TransformerLM, self).__init__(*layers)
