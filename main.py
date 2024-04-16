import argparse
import torch.distributed
import torch
import os
from utils import *
import psutil
import time

from Runtime import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_chunks', default=4, type=int, help='mumber of micro batches')
    parser.add_argument('--seq_length', default=128, type=int, help='sequence length')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--num_layers', default=24, type=int, help='number of Transformer layers in the whole model')
    parser.add_argument('--num_stages', default = 4, type=int, help='number of stages in the pipeline')
    parser.add_argument('--embedding_dim', default=1024, type=int, help='embedding dimension in a Transformer layer')
    parser.add_argument('--hidden_dim', default=4096, type=int, help='hidden dimension in a Transformer layer')
    parser.add_argument('--num_warmup_steps', default=2, type=int, help='number of iterations to warm up before training')
    parser.add_argument('--num_training_steps', default=2, type=int, help='number of iterations to time in experiment')
    parser.add_argument('--num_heads', default=32, type=int, help='number of attention heads per layer')
    parser.add_argument('--num_experts', default=8, type=int, help='num experts per layer')
    parser.add_argument('--pipeline', default="GPipe", choices=["GPipe", "GPipeOffload", "Mobius","APTMoE"], type=str, help='pipeline parallelism approach')
    parser.add_argument('--model_config', default="S", choices=["S", "M", "L"], type=str, help='model configuration')
    parser.add_argument('--is_moe', default=False, help='whether to use MoE models')
    parser.add_argument('--fwd_only', action='store_true', help='Only perform forward pass (three loading phases)')
    parser.add_argument('--inter_stage_only', action='store_true', help='Only perform inter-stage phase')
    parser.add_argument('--gini', default=0.6, type=float, help='expert workload imbalance degree')
    parser.add_argument('--topo', default='C1+G2', choices=['C1+G1', 'C1+G2', 'C1+G4'], type=str, help='device topology')

    args = parser.parse_args()
    args = model_config(args, args.model_config)
    
    args.inter_stage_only = True
    
    print(args)
    
    assert args.num_layers % args.num_stages == 0

    torch.distributed.init_process_group(backend='nccl') 
    set_seed(args.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_size = torch.cuda.device_count()

    print('Local rank %d, Global rank %d, Local size %d, World size %d.' % (local_rank, global_rank, local_size, world_size))
    
    torch.cuda.set_device(local_rank % local_size)

    cpu_count = psutil.cpu_count() 
    p = psutil.Process()
    
    if args.topo == 'C1+G2':
        cpu_num = cpu_count // local_size
    elif args.topo == 'C1+G4':
        cpu_num = cpu_count //2 // local_size
    elif args.topo == 'C1+G1':
        cpu_num = cpu_count 
        
    torch.set_num_threads(cpu_num)
    core_list = []
    for core_id in range(cpu_num * local_rank, cpu_num * (local_rank+1)):
        core_list.append(core_id)
    p.cpu_affinity(core_list) 
    print('global_rank = '+str(global_rank) + ' p.cpu_affinity() = '+str(p.cpu_affinity()))
    
      
    if args.pipeline == 'GPipe':
        action_list = generate_action_GPipe(world_size=world_size, num_chunks=args.num_chunks)[global_rank]
    elif args.pipeline == 'GPipeOffload':
        action_list = generate_action_GPipeOffload(world_size=world_size, num_stages=args.num_stages, num_chunks=args.num_chunks)[global_rank]
    else:
        action_list = generate_action_Mobius_APTMoE(world_size=world_size, num_stages=args.num_stages, num_chunks=args.num_chunks)[global_rank]
    
    
    comm_schedulers = [CommScheduler(device_id=i%local_size) for i in range(world_size)]
    R_solver = R_solver(args.model_config, args.num_experts, args.num_chunks)
    
    module_list = generate_pipeline(args, args.pipeline, world_size, local_size, comm_schedulers, R_solver)
    
    # print('rank = ' + str(global_rank) + ' action_list = '+str(action_list))

    PPRuntime = PipelineRuntime(batch_size=args.batch_size, num_chunks=args.num_chunks, seq_length=args.seq_length, model_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                          module_list=module_list, world_size=world_size, local_size=local_size, global_rank=global_rank, num_stages = args.num_stages, pipeline=args.pipeline, fwd_only=args.fwd_only)

   
    if local_rank == 0:
        print("--------------- begin warming up")
    for l in range(args.num_warmup_steps):
        PPRuntime.optimizer_.zero_grad()
        PPRuntime.run_pipeline(action_list=action_list)
        torch.distributed.barrier()
        PPRuntime.optimizer_.step()
    if local_rank == 0:
        print("--------------- finish warming up")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(args.num_training_steps):
        PPRuntime.optimizer_.zero_grad()
        PPRuntime.run_pipeline(action_list=action_list)
        torch.distributed.barrier()
        PPRuntime.optimizer_.step()
        if local_rank == 0:
            print(f"--------------- finish training step {i}")

    torch.cuda.synchronize()
    training_time = (time.time() - start_time) / args.num_training_steps
    
    verify_peak_memory(local_rank)
    if local_rank == 0:
        print(f"model #params = {PPRuntime.total_params:,}")
        print(f"training elapsed time per step = {training_time:.3f} s")
        print(f"training throughput = {(1/training_time)*args.batch_size:.3f}")


