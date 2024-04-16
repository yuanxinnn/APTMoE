from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import json

model_id = "./Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
total_layers = 32
hidden_inputs = []
gate_results = []

layer = 0
t_inputs = []
t_results = []
data_id = 0


def register(inp, gate_result):
    global layer, t_inputs, t_results, hidden_inputs, gate_results, data_id
    print(inp.shape, gate_result.shape)
    t_inputs.append(inp.cpu().squeeze(dim=0))
    gate_result = gate_result.float().cpu().squeeze(dim=0)
    t_results.append(gate_result)
    layer += 1
    if layer == total_layers:
        torch.save(np.array(t_inputs),
                   f"pths_mixtral/hidden_inputs.{data_id}.pth", pickle_protocol=4)
        torch.save(np.array(t_results),
                   f"pths_mixtral/gate_results.{data_id}.pth", pickle_protocol=4)
        t_inputs = []
        t_results = []
        layer = 0
        data_id += 1


model = AutoModelForCausalLM.from_pretrained(
    model_id, register, device_map="auto", revision="refs/pr/5")


def load_data():
    data_prefix = './data/'
    with open(data_prefix + "test.jsonl") as f:
        lines = f.readlines()
        questions = list(map(lambda x: json.loads(x)['solutions'], lines))
    with open(data_prefix + "train.jsonl") as f:
        lines = f.readlines()
        questions += (list(map(lambda x: json.loads(x)['solutions'], lines)))
    return questions


questions = load_data()

for i, question in enumerate(questions[:30]):
    print(f"iteration {i}...")
    inputs = tokenizer(question, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1)
    # print(outputs)
