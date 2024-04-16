import random
import torch
import numpy as np
from model.transformer_lm import TransformerLM
from Runtime import *
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def verify_peak_memory(rank):
    print(
        "cuda {:d}: Peak memory {:.2f} GB ; Persistent memory {:.2f} GB"
            .format(rank,torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"] / 2**30 , torch.cuda.memory_stats(rank)["allocated_bytes.all.current"] / 2**30)
    )


def model_config(args, model_conf):
    # MoE-S
    if model_conf == 'S':
        args.embedding_dim = 1024
        args.hidden_dim = 4096
        args.num_heads = 32
        args.num_layers = 32
        args.num_stages = 8
        args.batch_size = 128
        args.num_chunks = 4
        args.seq_length = 128
        args.prefetch_portion = 0.6
    # MoE-M
    elif model_conf == 'M':
        args.embedding_dim = 2048
        args.hidden_dim = 8192
        args.num_heads = 32
        args.num_layers = 32
        args.num_stages = 8
        args.batch_size = 16
        args.num_chunks = 4
        args.seq_length = 128  
        args.prefetch_portion = 0.7
    # MoE-L
    elif model_conf == 'L':
        args.embedding_dim = 4096
        args.hidden_dim = 14336
        args.num_heads = 32
        args.num_layers = 32
        args.num_stages = 8
        args.batch_size = 8
        args.num_chunks = 4
        args.seq_length = 128  
        args.prefetch_portion = 0.2
    return args


def generate_pipeline(args, pipeline, world_size, local_size, comm_schedulers, R_solver):
    module_list = []
    
    if pipeline == 'GPipe':
        args.num_stages = world_size
        layer_per_stage = args.num_layers//args.num_stages                                          
        for stage_rank in range(args.num_stages):
            module_list.append(TransformerLM(ninp=args.embedding_dim, nhead=args.num_heads, nhid=args.hidden_dim, dropout=0.0, 
                                            ndecoder=layer_per_stage, is_moe=args.is_moe, num_local_experts=args.num_experts, pipeline=args.pipeline, stage_id=stage_rank, gini=args.gini).cuda())
    
    elif pipeline == 'GPipeOffload':
        layer_per_stage = args.num_layers//args.num_stages 
        stage_per_device = args.num_stages // world_size
        for stage_rank in range(args.num_stages):
        
            compute_device_id = (stage_rank // stage_per_device) // (world_size // local_size)
            
            cur_stage = TransformerLM(ninp=args.embedding_dim, nhead=args.num_heads, nhid=args.hidden_dim, dropout=0.0, ndecoder=layer_per_stage, is_moe=args.is_moe, num_local_experts=args.num_experts, 
                                        pipeline=args.pipeline, stage_id=stage_rank, gini=args.gini, comm_scheduler=comm_schedulers[compute_device_id], R_solver=R_solver)
            
            module_list.append(
                ModelShard(
                model_shard=cur_stage,
                compute_device_id=compute_device_id,
                offload_device='cpu',
                index=stage_rank,
                offload_grained="coarse",
                )
            )
               
    elif pipeline == 'Mobius':
        layer_per_stage = args.num_layers//args.num_stages 
        for stage_rank in range(args.num_stages):
            compute_device_id = stage_rank % local_size
            
            cur_stage = TransformerLM(ninp=args.embedding_dim, nhead=args.num_heads, nhid=args.hidden_dim, dropout=0.0, ndecoder=layer_per_stage, is_moe=args.is_moe, num_local_experts=args.num_experts,
                                            pipeline=args.pipeline, stage_id=stage_rank, gini=args.gini, comm_scheduler=comm_schedulers[compute_device_id], R_solver=R_solver)
            module_list.append(
                ModelShard(
                model_shard=cur_stage,
                compute_device_id=compute_device_id,
                offload_device='cpu',
                index=stage_rank,
                offload_grained="coarse",
                )
            )
                
    
    elif pipeline == 'APTMoE':
        layer_per_stage = args.num_layers//args.num_stages 
        for stage_rank in range(args.num_stages):
            compute_device_id = stage_rank % local_size
            
            cur_stage = TransformerLM(ninp=args.embedding_dim, nhead=args.num_heads, nhid=args.hidden_dim, dropout=0.0, ndecoder=layer_per_stage, is_moe=args.is_moe, num_local_experts=args.num_experts,
                                            pipeline=args.pipeline, stage_id=stage_rank, gini=args.gini, comm_scheduler=comm_schedulers[compute_device_id], R_solver=R_solver, inter_stage_only=args.inter_stage_only)
            module_list.append(
                ModelShard(
                model_shard=cur_stage,
                compute_device_id=compute_device_id,
                offload_device='cpu',
                index=stage_rank,
                offload_grained="fine",
                inter_stage_only=args.inter_stage_only,
                )
            )  
    return module_list
        
