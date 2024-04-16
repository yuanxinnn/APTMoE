from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple

import torch
from torch import nn
import copy
import random


def clear_list(lst):
    for i in range(len(lst)):
        lst[i] = 0

prefetch_portion = 0.5

def random_stageload_list(num_experts, portion):
    numbers = list(range(num_experts))
    count = int(portion * num_experts)
    
    sampled_numbers = random.sample(numbers, count)
    sampled_numbers.sort()
    return sampled_numbers

   
class ModelShard(nn.Sequential):

    def __init__(
        self,
        model_shard: nn.Sequential,
        compute_device_id: int,
        offload_device: torch.device,
        index: int,
        offload_grained: str,
        inter_stage_only=None,
    ):
        super().__init__()
        self.model_shard = model_shard
        self.index = index
        self.offload_grained = offload_grained
        self.inter_stage_only = inter_stage_only

        self.compute_device = torch.device('cuda', compute_device_id)
        self.compute_device_id = compute_device_id
        
        self.CommScheduler = self.model_shard.comm_scheduler
        self.R_solver = self.model_shard.R_solver

        self.offload_device = offload_device

        self.model_shard.to(offload_device) 
        
        self.StageLoadEvent = torch.cuda.Event()
        self.StageCompEvent = torch.cuda.Event()
    

    def forward(self, inputs, chunk_id = None, nextModelShard = None):
        
        if self.offload_grained == 'coarse' or self.inter_stage_only:
            with torch.cuda.stream(self.model_shard.comp_stream):
                torch.cuda.current_stream().wait_event(self.StageLoadEvent)
                inputs =  self.model_shard(inputs)
                self.StageCompEvent.record(torch.cuda.current_stream())
        
        # Level2 (Inter-Layer)： for all layers of the current stage in the current micro batch, execute them layer-by-layer
        # When executing the previous layer, modify the predicted popularity of the next layer 
        # and prefetch the predicted experts of the next layer
        # For each micro batch, update the expert-to-device allcation scheme through adding/removing operation
        else:
            
            # self.CommScheduler.print_pq()
            
            # clear inter-stage queue in 0-th micro batch
            if chunk_id == 0:
                self.CommScheduler.clear_priority(2)
                
                # If some MHAs and Gates have not been loaded, then add them to the inter-layer queue
                for (layer_id, layer) in enumerate(self.model_shard):
                    if layer.self_attn.on_GPU == False:
                        MHA = []
                        MHA.append(layer.self_attn)
                        MHA.append(layer.norm1)
                        MHA.append(layer.norm2)
                        MHA.append(layer.dropout)
                        self.CommScheduler.add_model_to_queue(MHA, layer.MHA_event, 1)     
                        self.CommScheduler.load_execute_with_priority()
                    
                    if layer.moe_layer.gate.on_GPU == False:
                        Gate = []
                        Gate.append(layer.gate)
                        Gate.append(layer.next_gate)
                        self.CommScheduler.add_model_to_queue(Gate, layer.Gate_event, 1)
                        self.CommScheduler.load_execute_with_priority()

            
            expert_selection_prediction = None
            gpu_expert_ids_prev = None  
            
            for (layer_id, layer) in enumerate(self.model_shard):

                # inter-layer loading
                if expert_selection_prediction is not None:        
                    layer.moe_layer.predicted_expert_selection_list.append(expert_selection_prediction)
                    
                    predicted_assigned_tokens_list = layer.moe_layer.assigned_tokens_list
                    for i in range(len(expert_selection_prediction)):
                        predicted_assigned_tokens_list[i] += expert_selection_prediction[i]
                    
                    gpu_expert_ids_new = self.R_solver.solve(layer.moe_layer.assigned_tokens_list)
                    
                    if gpu_expert_ids_prev is not None:
                        gpu_expert_ids_to_remove = [i for i in gpu_expert_ids_prev if i not in gpu_expert_ids_new]
                        gpu_expert_ids_to_add = [i for i in gpu_expert_ids_new if i not in gpu_expert_ids_prev]
                    else:
                        gpu_expert_ids_to_remove = []
                        gpu_expert_ids_to_add = gpu_expert_ids_new

                    gpu_expert_ids_prev = gpu_expert_ids_new
                    
                    # print('gpu_expert_ids_to_remove = '+str(gpu_expert_ids_to_remove) + ' gpu_expert_ids_to_add = '+str(gpu_expert_ids_to_add))
                    
                    # add experts that are not in inter-layer queue but in the latest scheme
                    for expert_id in gpu_expert_ids_to_add:
                        if self.CommScheduler.contain_model(layer.moe_layer.experts[expert_id], 1) == False:
                            self.CommScheduler.add_model_to_queue(layer.moe_layer.experts[expert_id], layer.expert_events[expert_id], 1)

                    # remove experts that are in the inter-layer queue but not in the latest scheme
                    for expert_id in gpu_expert_ids_to_remove:
                        if self.CommScheduler.contain_model(layer.moe_layer.experts[expert_id], 1) == True:
                            self.CommScheduler.remove_model_from_queue(layer.moe_layer.experts[expert_id], layer.expert_events[expert_id], 1)
                    
                    self.CommScheduler.load_execute_with_priority()                   
                                   

                    with torch.cuda.stream(self.model_shard.comp_stream):
                        self.model_shard.comp_stream.wait_event(layer.MHA_event)
                        self.model_shard.comp_stream.wait_event(layer.Gate_event)
                        inputs = layer(inputs)
                          
                # the 0-th layer of tue current stage, skip inter-layer loading                    
                else: 
                    with torch.cuda.stream(self.model_shard.comp_stream):
                        inputs =  layer(inputs)
                
                expert_selection_prediction = layer.moe_layer.next_layer_expert_selection_prediction
                    
            self.StageCompEvent.record(self.model_shard.comp_stream)

        return inputs


    def FwdStageLoad(self, chunk_id = None, num_chunks = None) -> Any:
        
        # perform loading in the 0-th micro batch
        if self.offload_grained == "coarse":
            if chunk_id == 0:
                self.CommScheduler.load_execute(self.model_shard, waitEvent = None, recordEvent = self.StageLoadEvent) 

        elif self.inter_stage_only:
        
            if chunk_id == 0:
                fwd_load_model = []
                for (layer_id, layer) in enumerate(self.model_shard):

                    fwd_load_model.append(layer.self_attn)
                    fwd_load_model.append(layer.norm1)
                    fwd_load_model.append(layer.norm2)
                    fwd_load_model.append(layer.dropout)
                    fwd_load_model.append(layer.gate)
                    fwd_load_model.append(layer.next_gate)

                    fwd_gpu_expert_ids = self.R_solver.solve(layer.moe_layer.assigned_tokens_list)
                    
                    if len(fwd_gpu_expert_ids) == 0:
                        fwd_gpu_expert_ids = random_stageload_list(len(layer.moe_layer.experts), portion=prefetch_portion)
                    #     print('(random generated) fwd_gpu_expert_ids = '+str(fwd_gpu_expert_ids))
                    # else:
                    #     print('(global optimal) fwd_gpu_expert_ids = '+str(fwd_gpu_expert_ids))
                    
                    for expert_id in fwd_gpu_expert_ids:
                        fwd_load_model.append(layer.moe_layer.experts[expert_id])

                # print('stage '+str(self.model_shard.stage_id) +' FwdStageLoad' +' chunk '+str(chunk_id))
                self.CommScheduler.load_execute(fwd_load_model, waitEvent = None, recordEvent = self.StageLoadEvent)

        # Level1 (Inter-Stage)
        else:
            if chunk_id == 0:
                
                # first push MHAs and Gates of all layers with the highest demand
                for (layer_id, layer) in enumerate(self.model_shard):
                    
                    MHA = []
                    MHA.append(layer.self_attn)
                    MHA.append(layer.norm1)
                    MHA.append(layer.norm2)
                    MHA.append(layer.dropout)
                    
                    Gate = []
                    Gate.append(layer.gate)
                    Gate.append(layer.next_gate)
                    
                    self.CommScheduler.add_model_to_queue(MHA, layer.MHA_event, 2)
                    self.CommScheduler.add_model_to_queue(Gate, layer.Gate_event, 2)

                # then add the experts of all layers to the queue in descending order according to the historically popularity
                for (layer_id, layer) in enumerate(self.model_shard):
                    moe_layer = layer.moe_layer 
                    origin_list = [i for i in range(len(moe_layer.experts))]
                    
                    sorted_expert_ids = sorted(origin_list, key=lambda x: moe_layer.historical_assigned_tokens_list[x], reverse=True)

                    for expert_id in sorted_expert_ids:
                        self.CommScheduler.add_model_to_queue(moe_layer.experts[expert_id], layer.expert_events[expert_id], 2)
                            
                self.CommScheduler.load_execute_with_priority()


    def BwdStageLoad(self, chunk_id = None, num_chunks = None) -> Any:
        if self.offload_grained == "coarse":
            if chunk_id == 0:
                self.CommScheduler.load_execute(self.model_shard, waitEvent = None, recordEvent = self.StageLoadEvent) 

        else:
            # backward loading strategy: 
            # determine the global optimal scheme according to the exeprt popularity of the forward process
            if chunk_id == 0:
                bwd_load_model = []
                for (layer_id, layer) in enumerate(self.model_shard):

                    bwd_load_model.append(layer.self_attn)
                    bwd_load_model.append(layer.norm1)
                    bwd_load_model.append(layer.norm2)
                    bwd_load_model.append(layer.dropout)
                    bwd_load_model.append(layer.gate)
                    bwd_load_model.append(layer.next_gate)

                    bwd_gpu_expert_ids = self.R_solver.solve(layer.moe_layer.assigned_tokens_list)
                    
                    if len(bwd_gpu_expert_ids) == 0:
                        bwd_gpu_expert_ids = random_stageload_list(len(layer.moe_layer.experts), portion=prefetch_portion)
                        # print('(random generated) bwd_gpu_expert_ids = '+str(bwd_gpu_expert_ids))
                    # else:
                    #     print('(global optimal) bwd_gpu_expert_ids = '+str(bwd_gpu_expert_ids))
                    
                    for expert_id in bwd_gpu_expert_ids:
                        bwd_load_model.append(layer.moe_layer.experts[expert_id])

                # print('stage '+str(self.model_shard.stage_id) +' BwdStageLoad' +' chunk '+str(chunk_id))
                self.CommScheduler.load_execute(bwd_load_model, waitEvent = None, recordEvent = self.StageLoadEvent)


    def FwdStageDrop(self, chunk_id = None, num_chunks = None, fwd_only = False) -> Any:
            # drop all model blocks in the last micro batch
            if chunk_id == num_chunks - 1:
                self.CommScheduler.drop_execute(self.model_shard, waitEvent = self.StageCompEvent, recordEvent = None) 
                if fwd_only:
                    for layer in self.model_shard:
                        clear_list(layer.moe_layer.assigned_tokens_list)


    def BwdStageDrop(self, chunk_id = None, num_chunks = None) -> Any:
        if chunk_id == num_chunks - 1:
            # print('stage '+str(self.model_shard.stage_id) +' BwdStageDrop' +' chunk '+str(chunk_id))
            self.CommScheduler.drop_execute(self.model_shard, waitEvent = self.StageCompEvent, recordEvent = None) 
            for layer in self.model_shard:
                layer.moe_layer.historical_assigned_tokens_list = layer.moe_layer.assigned_tokens_list
                clear_list(layer.moe_layer.assigned_tokens_list)
