from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast, List

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList


if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module
    
import random

def generate_similiar_list(origin_list):
    sorted_origin = sorted(enumerate(origin_list), key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sorted_origin]

    target_sum = sum(origin_list)
    
    proportion_same = 0.3
    num_elements_same = int(len(origin_list) * proportion_same)
    list_same = random.sample(origin_list, k=num_elements_same)

    current_sum = sum(list_same)
    num_elements_to_add = len(origin_list) - num_elements_same
    new_elements = []
    target_remaining_sum = target_sum - current_sum

    while len(new_elements) < num_elements_to_add:
        new_element = random.randint(0, target_remaining_sum)

        if len(new_elements) + 1 < num_elements_to_add:
            new_elements.append(new_element)
            target_remaining_sum -= new_element
        else:
            new_elements.append(target_remaining_sum)
            break

    new_list_sorted_origin = list_same + new_elements
    new_list_sorted_origin.sort(reverse = True)
    
    new_list = [0] * len(origin_list)
    for i, val in enumerate(sorted_indices):
        new_list[val] = new_list_sorted_origin[i]

    return new_list


class MoELayer(Base):
    def __init__(self, gate: Module, next_gate: Module, layer_id: int, ndecoder: int, experts: Union[Module, ModuleList], 
                 pipeline: str = 'GPipe', comp_stream=None, comm_scheduler=None, R_solver=None, Gate_event=None, expert_events=None, inter_stage_only=None) -> None:
        super().__init__()
        self.gate = gate
        self.layer_id = layer_id
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
            
        self.num_local_experts = len(self.experts)
        self.next_gate = next_gate
        self.pipeline = pipeline
        self.predicted_expert_selection_list = []
        self.next_layer_expert_selection_prediction = None
        self.ndecoder = ndecoder
        self.comp_stream = comp_stream
        self.CommScheduler = comm_scheduler
        self.R_solver = R_solver
        self.Gate_event = Gate_event
        self.expert_events = expert_events
        self.inter_stage_only = inter_stage_only
        
        self.real_expert_selection = []
        self.assigned_tokens_list = [0] * len(self.experts)
        self.historical_assigned_tokens_list = [0] * len(self.experts)
        
        
    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[1] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        d_model = input[0].shape[2] 
        reshaped_input = input[0].reshape(-1, d_model)  
        expert_outputs = []              


        # GPipe GPipeOffload Mobius
        if self.pipeline == 'GPipe' or self.pipeline == 'GPipeOffload' or self.pipeline == 'Mobius' or self.inter_stage_only:
            self.real_expert_selection = self.gate(reshaped_input)
    
            chunks = torch.split(reshaped_input, self.real_expert_selection, dim=0)
               
            for chunk, expert in zip(chunks, self.experts):
                if chunk.shape[0] != 0:
                    expert_outputs += [expert(chunk)]
            
            if self.inter_stage_only:
                for i in range(len(self.experts)):
                    self.assigned_tokens_list[i] += self.real_expert_selection[i]


        elif self.pipeline == 'APTMoE':
            self.comp_stream.wait_event(self.Gate_event)
            
            self.real_expert_selection = self.gate(reshaped_input)
            
            # simulate the real expert selection as the predicted one
            if len(self.predicted_expert_selection_list) != 0:
                self.real_expert_selection = generate_similiar_list(self.predicted_expert_selection_list[-1])

            # assume the real expert selection as the predicted one
            # if len(self.predicted_expert_selection_list) != 0:
            #     self.real_expert_selection = self.predicted_expert_selection_list[-1]
            
            for i in range(len(self.experts)):
                self.assigned_tokens_list[i] += self.real_expert_selection[i]
                
            self.next_layer_expert_selection_prediction = self.next_gate(reshaped_input)

            # print(self.real_expert_selection)
            # print(sum(self.real_expert_selection))
        
            chunks = torch.split(reshaped_input, self.real_expert_selection, dim=0)
        
            # Level3 (Inter-Expert)：
            # accoirding to real-time popularity, add some experts to the highest priority queue
            
            gpu_expert_ids = self.R_solver.solve(self.assigned_tokens_list)
            
            for expert_id in gpu_expert_ids:
                if self.experts[expert_id].on_GPU == False:
                    self.CommScheduler.add_model_to_queue(self.experts[expert_id], self.expert_events[expert_id], 0)
            
            self.CommScheduler.load_execute_with_priority()

            gpu_execution_ids = []
            cpu_execution_ids = []
            
            for num_tokens, expert in zip(self.real_expert_selection, self.experts):
                if expert.on_GPU == True:
                    gpu_execution_ids.append(expert.expert_id)
                else:
                    cpu_execution_ids.append(expert.expert_id)
    
            # execute experts on GPU
            with torch.cuda.stream(self.comp_stream):        
                for id in gpu_execution_ids:
                    if chunks[id].shape[0] != 0:
                        self.comp_stream.wait_event(self.expert_events[id])
                        expert_outputs += self.experts[id](chunks[id])
                        
            # execute experts on CPU
            for id in cpu_execution_ids:
                if chunks[id].shape[0] != 0:
                    expert_outputs += self.experts[id](chunks[id])

        
        expert_output = torch.cat(expert_outputs, dim=0)
        expert_output = expert_output.reshape(input[0].shape)

        return expert_output
   