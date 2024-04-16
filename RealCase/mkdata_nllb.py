from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
import json

total_layers = 6
hidden_inputs = []
gate_results = []

layer = 0
t_inputs = []
t_results = []
register_cnt = 0
data_id = 0


def register(inp, gate_result):
    global layer, t_inputs, t_results, hidden_inputs, gate_results, data_id, register_cnt
    print(inp.shape, gate_result.shape)
    t_inputs.append(inp.cpu().squeeze(dim=0))
    gate_result = gate_result.float().cpu().squeeze(dim=0)
    t_results.append(gate_result)
    layer += 1
    if layer == total_layers:
        layer = 0
        if register_cnt % 2 == 1:
            register_cnt += 1
            t_inputs.clear()
            t_results.clear()
            return
        register_cnt += 1
        torch.save(np.array(t_inputs),
                   f"pths_nllb/hidden_inputs.{data_id}.pth", pickle_protocol=4)
        torch.save(np.array(t_results),
                   f"pths_nllb/gate_results.{data_id}.pth", pickle_protocol=4)
        t_inputs = []
        t_results = []
        data_id += 1


tokenizer = AutoTokenizer.from_pretrained(
    "./nllb-moe-54b", device_map="auto", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./nllb-moe-54b", register, device_map="auto", local_files_only=True)


def load_data():
    data_prefix = './data/'
    with open(data_prefix + "test.jsonl") as f:
        lines = f.readlines()
        questions = list(map(lambda x: json.loads(x)['question'], lines))
    with open(data_prefix + "train.jsonl") as f:
        lines = f.readlines()
        questions += (list(map(lambda x: json.loads(x)['question'], lines)))
    return questions


questions = load_data()

for i, question in enumerate(questions[:200]):
    print(f"iteration {i}...")
    inputs = tokenizer(question, return_tensors="pt")
    inputs = inputs.to("cuda")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=1
    )
    # print(outputs)
