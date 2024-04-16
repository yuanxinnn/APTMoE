import torch
import queue
from model.transformer_lm import ExpertFeedForwardLayer as ExpertType
from model.transformer_lm import Attention as MHAType
from model.top2gate import RandomGate as GateType
import torch.nn as nn


class PriorityQueue:
    def __init__(self):
        self.queues = {0: queue.Queue(), 1: queue.Queue(), 2: queue.Queue()}

    def add_task(self, task, priority):
        if priority in self.queues:
            self.queues[priority].put(task)
        else:
            raise ValueError("Invalid Priority. (0, 1 or 2)")

    def remove_task(self, task):
        for priority in self.queues:
            q = self.queues[priority]
            with q.mutex:
                q.queue = deque([item for item in q.queue if item != task])

    def pop_task(self):
        for priority in range(0, 3):
            if not self.queues[priority].empty():
                task = self.queues[priority].get()
                model = task[0]
                event = task[1]
                return priority, model, event
        return -1, None, None

    def clear_priority(self, priority):
        self.queues[priority] = queue.Queue()

    def contain_task(self, task, priority):
        return task in list(self.queues[priority].queue)

    def print_pq(self):
        print('print priority queue:')
        for priority in range(0, 3):
            tasks_count = len(self.queues[priority].queue)
            if tasks_count != 0:
                print(f"Priority {priority} tasks count: {tasks_count}")


class CommScheduler:
    def __init__(self, device_id):
        super().__init__()
        self.load_stream = torch.cuda.Stream()
        self.drop_stream = torch.cuda.Stream()
        
        self.pq = PriorityQueue()
        self.device_id = device_id


    def load(self, m):
        m.to(torch.device('cuda', self.device_id), non_blocking = True)
        if isinstance(m, ExpertType) or isinstance(m, MHAType) or isinstance(m, GateType):
            m.on_GPU = True 


    def drop(self, m):
        m.to('cpu', non_blocking = True)
        if isinstance(m, ExpertType) or isinstance(m, MHAType) or isinstance(m, GateType):
            m.on_GPU = False 


    def load_execute_with_priority(self):
        priority, model, event = self.pq.pop_task()
        
        if model is not None:
            # print('load_execute_with_priority = '+str(priority))
            
            with torch.cuda.stream(self.load_stream):
                
                if isinstance(model, list):
                    for m in model:
                        self.load(m)
                else:
                    self.load(model)
                
                event.record(torch.cuda.current_stream())
                if event.query():
                    self.load_execute_with_priority()
    
    
    def add_model_to_queue(self, model, event, priotiry):
        self.pq.add_task((model, event), priotiry)

    def remove_model_from_queue(self, model, event, priotiry):
        self.pq.remove_task((model, event))

    def print_pq(self):
        self.pq.print_pq()

    def contain_model(self, model, priority):
        self.pq.contain_task(model, priority)
    
    def clear_priority(self, priority):
        self.pq.clear_priority(priority)

    def clear_queues(self):
        for priority in range(0, 3):
            self.pq.clear_priority(priority)

    def load_execute(self, model, waitEvent = None, recordEvent = None):
        with torch.cuda.stream(self.load_stream):
            if waitEvent is not None:
                torch.cuda.current_stream().wait_event(waitEvent)
            
            if isinstance(model, list):
                for m in model:
                    self.load(m)
            else:
                model.to(torch.device('cuda', self.device_id), non_blocking = True)
                for layer in model:
                    layer.self_attn.on_GPU = True
                    layer.moe_layer.gate.on_GPU = True
                    for expert in layer.moe_layer.experts:
                        expert.on_GPU = True
                
            if recordEvent is not None:
                recordEvent.record(torch.cuda.current_stream())

    def drop_execute(self, model, waitEvent = None, recordEvent = None):
        with torch.cuda.stream(self.drop_stream):
            if waitEvent is not None:
                torch.cuda.current_stream().wait_event(waitEvent)
                
            model.to('cpu', non_blocking = True)

            for layer in model:
                layer.self_attn.on_GPU = False
                layer.moe_layer.gate.on_GPU = False
                for expert in layer.moe_layer.experts:
                    expert.on_GPU = False
                
            if recordEvent is not None:
                recordEvent.record(torch.cuda.current_stream())
