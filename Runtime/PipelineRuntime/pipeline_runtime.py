import torch
import torch.distributed as dist
from typing import Dict, List, Tuple

from ..OffloadRuntime import *


class PipelineRuntime():
    def __init__(self, batch_size: int, num_chunks: int, seq_length: int, model_dim: int,  hidden_dim: int, module_list: list,
                world_size: int, local_size: int, global_rank: int, num_stages: int, pipeline: str, fwd_only: bool):
        self.world_size = world_size
        self.local_size = local_size
        self.global_rank = global_rank
        self.num_stages = num_stages
        self.last_req = None
        self.last_req_backward = None
        self.batch_activation_list = [[] for _ in range(len(module_list))]
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.seq_length = seq_length
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.d_type = torch.float32
        self.module_list = module_list
        self.input_itr = 0
        self.total_params = 0
        self.input_src_list = [[] for _ in range(len(module_list))]
        self.input_batch_list = [[] for _ in range(len(module_list))]
        self.input_grad_list = [[] for _ in range(len(module_list))]
        self.self_send_backward_tensor = None
        self.last_req_recv = None
        self.last_tensor_recv = None
        self.optimizer_ = self.constructing_optimizer_and_count_params()
        self.pipeline = pipeline
        self.fwd_only = fwd_only
        
    def constructing_optimizer_and_count_params(self):
        params = []

        if isinstance(self.module_list[0], ModelShard):
            for module in self.module_list:
                model_params = list(module.model_shard.parameters())
                params += model_params
                self.total_params += sum(p.numel() for p in model_params)
        else:
            for module in self.module_list:
                model_params = list(module.parameters())
                params += model_params
                self.total_params += sum(p.numel() for p in model_params)
        return torch.optim.Adam(params, lr=0.00015, weight_decay=1e-2)


    def fetch_data(self):
        """fetch the batches of data needed in this accumulation"""
        self.input_itr = 0
        self.input_src_list[0] = []
        for i in range(0, self.world_size):
            input_tensor = torch.rand(self.batch_size//self.num_chunks, self.seq_length, self.model_dim).requires_grad_(True).to("cuda")
            self.input_src_list[0].append(input_tensor)


    def parse_action(self, action: str) -> tuple:
        """The format should be '{action} {local_module_rank} {target_rank}'
           While the action should be within [forward, backward, send_tensor, load_forward, fetch_data, send_grad],
           local_module_rank should be the rank of the module involved in this action within this node,
           and the target_rank should be the rank number of the target node, which should
           be an integer.
           The parameter should be set to -1 if not related to this action.
           Example: ['fetch_data -1 -1', 'load_forward 0 -1', 'send_tensor 0 1', 'forward 3 1', 'backward 3 1', 'backward 0 1']"""
        assert action.count(" ") == 3, 'There should be four arguments in an action'
        act_name = action.split(' ')[0]
        module_rank = int(action.split(' ')[1])
        tar_rank = int(action.split(' ')[2])
        num_chunk = int(action.split(' ')[3])
        return act_name, module_rank, tar_rank, num_chunk


    def recv_tensor(self, src: int, mod_rank: int, forward=True) -> torch.tensor:
        """receiving the tensor from the src node"""

        req1 = None
        recv_tensor_ = torch.zeros([self.batch_size//self.num_chunks, self.seq_length, self.model_dim], dtype=self.d_type, device="cuda")
        
        if self.pipeline == 'GPipeOffload':
            stage_per_device = self.num_stages // self.world_size
            src_device_id = (src // stage_per_device) // (self.world_size // self.local_size)   
            if src_device_id != self.global_rank:
                req1 = dist.irecv(src=src_device_id, tensor=recv_tensor_)
        else:
            src_device_id = src % self.world_size
            req1 = dist.irecv(src=src_device_id, tensor=recv_tensor_)
        
        if forward:
            recv_tensor_.requires_grad = True

        return recv_tensor_, req1


    def send_tensor(self, mod_rank: int, target: int):
        """Wait for the last send request to finish and start new request"""

        if self.last_req is not None:
            self.last_req.wait()
            # torch.cuda.synchronize()
        
        req = None
        tensor_to_be_sent = self.batch_activation_list[mod_rank][-1]
        if self.pipeline == 'GPipeOffload':
            stage_per_device = self.num_stages // self.world_size
            dst_device_id = (target // stage_per_device) // (self.world_size // self.local_size)
            if dst_device_id != self.global_rank:
                req = dist.isend(tensor_to_be_sent, dst=dst_device_id)
        else:
            dst_device_id = target%self.world_size
            req = dist.isend(tensor_to_be_sent, dst=dst_device_id)
        self.last_req = req
        return


    def send_grad(self, mod_rank: int, target: int):
        """Wait for the last send_backward request to finish and start new request"""
        if self.last_req_backward is not None:
            self.last_req_backward.wait()
        if len(self.input_grad_list[mod_rank]) == 0:
            print(f'self.input_grad_list[{mod_rank}] == 0')
        temp_gr = self.input_grad_list[mod_rank].pop(0)
        
        req = None
        if self.pipeline == 'GPipeOffload':
            stage_per_device = self.num_stages // self.world_size
            dst_device_id = (target // stage_per_device) // (self.world_size // self.local_size)
            if dst_device_id != self.global_rank:
                req = dist.isend(temp_gr, dst=dst_device_id)    
        else:
            dst_device_id = target%self.world_size
            req = dist.isend(temp_gr, dst=dst_device_id)
        
        self.last_req_backward = req
        return


    def load_data(self):
        input_ = self.input_src_list[0][self.input_itr]
        self.input_itr += 1
        return input_


    def forward_pass(self, mod_rank: int, source_tensor: torch.tensor, chunk_id: int):
        """Pass the tensor through a certain module"""
        
        target_module = self.module_list[mod_rank]
        if mod_rank != self.num_stages -1:
            next_module = self.module_list[mod_rank+1]
        else:
            next_module = None
            
        if self.pipeline == 'GPipe':
            self.input_batch_list[mod_rank].append(source_tensor)
            res_tensor = target_module(source_tensor)
            
            
        elif self.pipeline == 'GPipeOffload':        

            target_module.FwdStageLoad(chunk_id, self.num_chunks)
            
            # prefetch next stage
            if mod_rank + 1 < len(self.module_list):
                next_target_module = self.module_list[mod_rank + 1]
                next_target_module.FwdStageLoad(chunk_id, self.num_chunks)
            
            self.input_batch_list[mod_rank].append(source_tensor)
            res_tensor = target_module(source_tensor, chunk_id, next_module)
        
            target_module.FwdStageDrop(chunk_id, self.num_chunks, self.fwd_only)  
        
        
        elif self.pipeline == 'Mobius' or self.pipeline == 'APTMoE':
    
            target_module.FwdStageLoad(chunk_id, self.num_chunks)
            # prefetch next stage
            if mod_rank + self.world_size < len(self.module_list):
                next_target_module = self.module_list[mod_rank + self.world_size]
                next_target_module.FwdStageLoad(chunk_id, self.num_chunks)
            
            self.input_batch_list[mod_rank].append(source_tensor)
            res_tensor = target_module(source_tensor, chunk_id, next_module)
        
            target_module.FwdStageDrop(chunk_id, self.num_chunks, self.fwd_only)

        
        self.batch_activation_list[mod_rank].append(res_tensor)
        
        return


    def backward_pass(self, mod_rank: int, output_grad: torch.tensor, src: int, chunk_id:int):
        input_tensor = self.input_batch_list[mod_rank].pop(0)
        output_tensor = self.batch_activation_list[mod_rank].pop(0)
        
        if self.fwd_only == False:
            if input_tensor.requires_grad:
                input_tensor.retain_grad()
            
            if self.pipeline == 'GPipe':
                if src != self.global_rank:
                    if output_grad is None:
                        torch.autograd.backward(output_tensor.mean())
                    else:
                        torch.autograd.backward(output_tensor, grad_tensors=output_grad) 
            
            
            elif self.pipeline == 'GPipeOffload':  

                target_module = self.module_list[mod_rank]
                target_module.BwdStageLoad(chunk_id, self.num_chunks)
                
                # prefetch "previous" stage
                if mod_rank - 1 >= 0:
                    prev_target_module = self.module_list[mod_rank - 1]
                    prev_target_module.BwdStageLoad(chunk_id, self.num_chunks)
                
                with torch.cuda.stream(target_module.model_shard.comp_stream):
                    torch.cuda.current_stream().wait_event(target_module.StageLoadEvent)
                    if src != self.global_rank:
                        if output_grad is None:
                            torch.autograd.backward(output_tensor.mean())
                        else:
                            torch.autograd.backward(output_tensor, grad_tensors=output_grad)
                    target_module.StageCompEvent.record(torch.cuda.current_stream())
                
                target_module.BwdStageDrop(chunk_id, self.num_chunks)      
            
            
            elif self.pipeline == 'Mobius' or self.pipeline == 'APTMoE':
            
                target_module = self.module_list[mod_rank]
                target_module.BwdStageLoad(chunk_id, self.num_chunks)

                if mod_rank - self.world_size >= 0:
                    prev_target_module = self.module_list[mod_rank - self.world_size]
                    prev_target_module.BwdStageLoad(chunk_id, self.num_chunks)
                
                with torch.cuda.stream(target_module.model_shard.comp_stream):
                    torch.cuda.current_stream().wait_event(target_module.StageLoadEvent)
                    # print('stage '+str(mod_rank) + ' bwd comp ' + ' chunk '+str(chunk_id))
                    if src != self.global_rank:
                        if output_grad is None:
                            torch.autograd.backward(output_tensor.mean())
                        else:
                            torch.autograd.backward(output_tensor, grad_tensors=output_grad)
                    target_module.StageCompEvent.record(torch.cuda.current_stream())
                
                target_module.BwdStageDrop(chunk_id, self.num_chunks)
        
            if input_tensor.grad is not None:
                self.input_grad_list[mod_rank].append(input_tensor.grad)
                
        return


    def run_pipeline(self, action_list: list):
        for i in range(len(action_list)):
            act = action_list[i]
            
            action, module_rank, tar_rank, chunk_id = self.parse_action(act)  # The action and related node rank
            
            if action == 'forward':
                if self.last_req_recv is None and self.last_tensor_recv is None:
                    source_tensor, last_recv_req = self.recv_tensor(tar_rank, module_rank, forward=True)
                    self.last_tensor_recv = source_tensor
                    self.last_req_recv = last_recv_req
                if self.last_req_recv is not None:
                    self.last_req_recv.wait()
                source_tensor = self.last_tensor_recv.clone()
                self.last_tensor_recv = None
                self.last_req_recv = None
                source_tensor.retain_grad()                    
                self.forward_pass(module_rank, source_tensor, chunk_id)

            elif action == 'backward':
                if self.fwd_only == False:
                    if tar_rank == -1:
                        # this is the last stage for the whole pipeline
                        temp_grad = None
                    else:
                        if self.last_req_recv is None and self.last_tensor_recv is None and tar_rank != self.global_rank:
                            next_tensor, last_recv_req = self.recv_tensor(tar_rank, module_rank, forward=False)
                            self.last_tensor_recv = next_tensor
                            self.last_req_recv = last_recv_req
                        if self.last_req_recv is not None:
                            self.last_req_recv.wait()
                        if tar_rank != self.global_rank:
                            temp_grad = self.last_tensor_recv.clone()
                        else:
                            temp_grad = None
                        self.last_tensor_recv = None
                        self.last_req_recv = None

                    self.backward_pass(module_rank, temp_grad, tar_rank, chunk_id)
                
                else:
                    temp_grad = None
                    self.backward_pass(module_rank, temp_grad, tar_rank, chunk_id)
                
            elif action == "load_forward":
                input_tensor = self.load_data()
                self.forward_pass(module_rank, input_tensor, chunk_id)
                
            elif action == "fetch_data":
                self.fetch_data()
                
            elif action == "send_grad":
                if self.fwd_only == False:
                    self.send_grad(module_rank, tar_rank)
                else:
                    pass
                
            elif action == "send_tensor":
                self.send_tensor(module_rank, tar_rank)

            else:
                print("unknown action")
                exit()
            
            # print(act + " on node " + str(self.global_rank))
                

