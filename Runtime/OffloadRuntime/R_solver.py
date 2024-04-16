from Static import LookupTable_S, LookupTable_M, LookupTable_L

class R_solver():
    def __init__(self, model_config, num_experts, num_chunks):
        if model_config == 'S':
            self.lookup_table = LookupTable_S
        elif model_config == 'M':
            self.lookup_table = LookupTable_M
        elif model_config == 'L':
            self.lookup_table = LookupTable_L
        
        self.num_experts = num_experts
        self.num_chunks = num_chunks
    
    def solve(self, assigned_tokens_list):

        origin_list = [i for i in range(len(assigned_tokens_list))]
        sorted_list = sorted(origin_list, key=lambda x: assigned_tokens_list[x], reverse=False)
        
        cpu_expert_ids = []
        
        comp = 0
        if self.num_experts == 8:
            load = self.lookup_table['load_MHA'] + self.lookup_table['load_Gate_8_experts']
        elif self.num_experts == 16:
            load = self.lookup_table['load_MHA'] + self.lookup_table['load_Gate_16_experts']
        elif self.num_experts == 64:
            load = self.lookup_table['load_MHA'] + self.lookup_table['load_Gate_64_experts']
            
        
        for expert_id in sorted_list:
            num_token = assigned_tokens_list[expert_id]
            if num_token > len(self.lookup_table['comp_cpu']):
                num_token = num_token // self.num_chunks  # TODO
            comp += self.lookup_table['comp_cpu'][num_token]
            load += self.lookup_table['load_expert']
            # print(comp / load)
            if comp / load < 1 :
                cpu_expert_ids.append(expert_id)
            else:
                break
            
        gpu_expert_ids = [i for i in sorted_list if i not in cpu_expert_ids]
        
        # print(origin_list)
        # print(sorted_list)
        # print('cpu_expert_ids ' + str(cpu_expert_ids))
        # print('gpu_expert_ids ' + str(gpu_expert_ids))
        
        return gpu_expert_ids
