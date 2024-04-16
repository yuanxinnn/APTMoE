def parse_action_(action: str) -> tuple:
    """The format should be '{action} {local_stage_rank} {target_rank}'
       While the action should be within [forward, backward, send_tensor, load_forward, fetch_data, send_grad],
       local_stage_rank should be the rank of the module involved in this action within this node,
       and the target_rank should be the rank number of the target node, which should
       be an integer.
       The parameter should be set to -1 if not related to this action.
       Example: ['fetch_data -1 -1', 'load_forward 0 -1', 'send_tensor 0 1', 'forward 3 1', 'backward 3 1', 'backward 0 1']"""
    assert action.count(" ") == 2, 'There should be three arguments in an action'
    act_name = action.split(' ')[0]
    stage_rank = int(action.split(' ')[1])
    tar_rank = int(action.split(' ')[2])
    return act_name, stage_rank, tar_rank


def generate_action_GPipe(world_size: int, num_chunks: int)-> list:
    res_list = [[] for _ in range(world_size)]
    
    res_list[0].append("fetch_data -1 -1 -1")
    
    for stage_rank in range(world_size):
        last_stage_rank = stage_rank - 1
        next_stage_rank = stage_rank + 1 

        if stage_rank == 0:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("load_forward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))

        elif stage_rank == world_size - 1:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("backward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                   

        else:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
                res_list[stage_rank].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))  
                res_list[stage_rank].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                
    return res_list



def generate_action_Mobius_APTMoE(world_size: int, num_stages: int, num_chunks: int)-> list:
    res_list = [[] for _ in range(world_size)]
    
    res_list[0].append("fetch_data -1 -1 -1")
    
    # generate forward action list
    for stage_rank in range(num_stages):
        last_stage_rank = stage_rank - 1
        next_stage_rank = stage_rank + 1 

        if stage_rank == 0:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("load_forward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
        elif stage_rank == num_stages - 1:
            for chunk_id in range(num_chunks):
                res_list[stage_rank%world_size].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                
        else:
            for chunk_id in range(num_chunks):
                res_list[stage_rank%world_size].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
                res_list[stage_rank%world_size].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))       

    # generate backward action list (reverse)
    for stage_rank in range(num_stages-1, -1, -1):
        last_stage_rank = stage_rank - 1
        next_stage_rank = stage_rank + 1 

        if stage_rank == 0:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
        elif stage_rank == num_stages - 1:
            for chunk_id in range(num_chunks):
                res_list[stage_rank%world_size].append("backward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank%world_size].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                   
        else:
            for chunk_id in range(num_chunks):
                res_list[stage_rank%world_size].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))  
                res_list[stage_rank%world_size].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
                
    return res_list


def generate_action_GPipeOffload(world_size: int, num_stages: int, num_chunks: int)-> list:
    stages_per_device = num_stages // world_size
    res_list = [[] for _ in range(world_size)]
    
    res_list[0].append("fetch_data -1 -1 -1")
    
    # generate forward action list
    for stage_rank in range(num_stages):
        last_stage_rank = stage_rank - 1
        next_stage_rank = stage_rank + 1 

        if stage_rank == 0:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("load_forward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
        elif stage_rank == num_stages - 1:
            for chunk_id in range(num_chunks):
                res_list[stage_rank // stages_per_device].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                
        else:
            for chunk_id in range(num_chunks):
                res_list[stage_rank // stages_per_device].append("forward {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
                res_list[stage_rank // stages_per_device].append("send_tensor {} {} {}".format(stage_rank, next_stage_rank, chunk_id))       

    # generate backward action list (reverse)
    for stage_rank in range(num_stages-1, -1, -1):
        last_stage_rank = stage_rank - 1
        next_stage_rank = stage_rank + 1 

        if stage_rank == 0:
            for chunk_id in range(num_chunks):
                res_list[stage_rank].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))
        elif stage_rank == num_stages - 1:
            for chunk_id in range(num_chunks):
                res_list[stage_rank // stages_per_device].append("backward {} -1 {}".format(stage_rank, chunk_id))  
                res_list[stage_rank // stages_per_device].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))                   
        else:
            for chunk_id in range(num_chunks):
                res_list[stage_rank // stages_per_device].append("backward {} {} {}".format(stage_rank, next_stage_rank, chunk_id))  
                res_list[stage_rank // stages_per_device].append("send_grad {} {} {}".format(stage_rank, last_stage_rank, chunk_id))
                
    return res_list


if __name__ == "__main__":
    
    res = generate_action_GPipe(world_size=4, num_chunks=4)
    print('GPipe')
    for i in res:
        print(i)
    res = generate_action_GPipeOffload(world_size=4, num_stages=8, num_chunks=4)
    print('\nGPipeOffload')
    for i in res:
        print(i)
    res = generate_action_Mobius_APTMoE(world_size=4, num_stages=8, num_chunks=4)
    print('\nMobius and APTMoE')
    for i in res:
        print(i)