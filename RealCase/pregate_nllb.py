from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import json
import argparse
import time

class PreGate(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts)
        
    
    def forward(self, x):
        # x: 1, 12, 4096 -> 1, 12, 8
        gate_result = self.gate(x)
        return gate_result

pregate_device = "cuda:0"
total_layers = 6

def load_pth(data_id: int):
    hidden_inputs = torch.from_numpy(torch.load(f"./pths_nllb/hidden_inputs.{data_id}.pth"))
    gate_results = torch.from_numpy(torch.load(f"./pths_nllb/gate_results.{data_id}.pth"))
    print(hidden_inputs.shape, gate_results.shape)
    return hidden_inputs, gate_results

training_data = []
testing_data = []
for i in range(200):
    print("load data", i, "...")
    if i < 180:
        training_data.append(load_pth(i))
    else:
        testing_data.append(load_pth(i))
training_data_X = torch.concatenate([data[0] for data in training_data], dim=1)
training_data_y = torch.concatenate([data[1] for data in training_data], dim=1)
testing_data_X = torch.concatenate([data[0] for data in testing_data], dim=1)
testing_data_y = torch.concatenate([data[1] for data in testing_data], dim=1)
print(training_data_X.shape, training_data_y.shape)
print(testing_data_X.shape, testing_data_y.shape)


class dataset(torch.utils.data.Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y
    def __len__(self):
        return self.data.shape[1]
    def __getitem__(self, idx):
        return self.data[:, idx, :], self.y[:, idx, :]
    
training_dataset = dataset(training_data_X, training_data_y)
testing_dataset = dataset(testing_data_X, testing_data_y)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1024, shuffle=True)

parser = argparse.ArgumentParser()
parser.add_argument('--head', type=int, default=1, help='Number of headout layers')
args = parser.parse_args()

headout_layers_num = args.head

num_experts = 128
hidden_dim = 2048
pre_gates = [PreGate(hidden_dim, num_experts).to(pregate_device) for i in range(total_layers-headout_layers_num)]
optimizer = [torch.optim.Adam(pre_gate.parameters(), lr=0.001) for pre_gate in pre_gates]

def step(data, y):
    X = data.permute((1, 0, 2)).to(pregate_device)
    y = y.permute((1, 0, 2)).to(pregate_device)
    sum_loss = 0
    for i in range(total_layers-headout_layers_num):
        X_i = X[i, :, :]
        y_i = y[i+headout_layers_num, :, :]
        # print(X_i.shape, y_i.shape)
        pre_gate = pre_gates[i]
        gate_result = pre_gate(X_i)
        loss = F.cross_entropy(gate_result, y_i)
        loss.requires_grad_(True)
        optimizer[i].zero_grad()
        loss.backward()
        optimizer[i].step()
        sum_loss += loss.item()
        # print(f"Layer {i} loss {loss}")
    # print(f"sum loss {sum_loss/(total_layers-1)}")
    pass

def get_onehot(routing_weights):
    routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
    # print(expert_mask.shape)
    expert_mask = expert_mask.sum(dim=1)
    return expert_mask

def calcAcc():
    cnt_per_layer_0 = [0 for i in range(total_layers-1)]
    cnt_per_layer_1 = [0 for i in range(total_layers-1)]
    cnt_per_layer_2 = [0 for i in range(total_layers-1)]
    all_tokens = 0
    last_predict_acc = [0 for i in range(16)]
    last_predict_all = [0 for i in range(16)]
    expert_32_in = [0 for i in range(5)]
    num_test_epoch = 0
    for i, (data, y) in enumerate(testing_loader):
        inputs = data.permute((1, 0, 2)).to(pregate_device)
        y = y.permute((1, 0, 2)).to(pregate_device)
        all_tokens += inputs.shape[1]
        num_test_epoch += 1
        for layer in range(total_layers-headout_layers_num):
            evaluation_num_experts = [35, 40, 45, 48, 64]
            tokens_count_predict = torch.zeros((1, num_experts)).to(pregate_device)
            tokens_count_fact = torch.zeros((1, num_experts)).to(pregate_device)
            inputs_i = inputs[layer, :, :]
            results_i = y[layer+headout_layers_num, :, :]
            # print(f"Layer {layer}...")
            routing_weights = pre_gates[layer](inputs_i)
            ans_onehot = get_onehot(routing_weights)
            # print(ans_onehot.shape)
            tokens_count_predict += ans_onehot.sum(dim=0)
            fact_onehot = get_onehot(results_i)
            tokens_count_fact += fact_onehot.sum(dim=0)

            matched = ans_onehot * fact_onehot
            # print(matched[0])
            summed_matched = matched.sum(dim=-1)
            cnt_per_layer_0[layer] += (summed_matched == 0).sum().item()
            cnt_per_layer_1[layer] += (summed_matched == 1).sum().item()
            cnt_per_layer_2[layer] += (summed_matched == 2).sum().item()
            # print(f"Layer {layer} tokens count predict:")
            # print(tokens_count_predict)
            sorted_indices_predict = torch.argsort(tokens_count_predict, descending=True, stable=True)
            # print(f"Layer {layer} tokens count fact")
            # print(tokens_count_fact)
            sorted_indices_fact = torch.argsort(tokens_count_fact, descending=True, stable=True)
            if layer == total_layers-headout_layers_num-1:
                # print(f"Layer {layer}...")
                for i in range(0, 16):
                    last_predict = sorted_indices_predict[0, num_experts-(i+1)*8:num_experts]
                    last_fact = sorted_indices_fact[0, num_experts-(i+1)*8:num_experts]
                    last_predict_acc[i] += np.intersect1d(last_predict.cpu().numpy(), last_fact.cpu().numpy()).shape[0]
                    last_predict_all[i] += 8 * (i+1)
                for idx, last_eva_expert_num in enumerate(evaluation_num_experts):
                    expert_32_in[idx] += np.intersect1d(sorted_indices_predict[0, num_experts-32:num_experts].cpu().numpy(), sorted_indices_fact[0, num_experts-last_eva_expert_num:num_experts].cpu().numpy()).shape[0]
    for i in range(0, 16):
        print(f"last {(i+1) * 8} expert predict acc: {last_predict_acc[i]}/{last_predict_all[i]}, {last_predict_acc[i]/last_predict_all[i]}")
    for idx, last_eva_expert_num in enumerate(evaluation_num_experts):
        print(f"last 32 expert in {last_eva_expert_num} predict acc: {expert_32_in[idx]}/{32*num_test_epoch}, {expert_32_in[idx]/(32*num_test_epoch)}")
    

    print("all acc:")
    print(f"0: {sum(cnt_per_layer_0)/(total_layers-headout_layers_num) / all_tokens}")
    print(f"1: {sum(cnt_per_layer_1)/(total_layers-headout_layers_num) / all_tokens}")
    print(f"2: {sum(cnt_per_layer_2)/(total_layers-headout_layers_num) / all_tokens}")
    print(f"expectation: {(sum(cnt_per_layer_1) + 2*sum(cnt_per_layer_2))/(total_layers-headout_layers_num) / all_tokens}")

    pass

def display_shape():
    for i in range(30):
        print(f"Data {i}...")
        X, y = load_pth(i)
        print(X.shape, y.shape)

num_epoch = 400
sttime = time.time()
calctime = 0

num_steps_passed = 0
def train():
    for epoch in range(num_epoch):
        for i, (data, y) in enumerate(training_loader):
            global num_steps_passed
            step(data, y)
            if num_steps_passed % 10 == 0:
                global sttime, calctime
                edtime = time.time()
                calctime += edtime-sttime
                print(f"Time: {edtime-sttime}")
                print(f"epoch {epoch} iteration {i}, all steps {num_steps_passed}...Evaluation!")
                print(f"===================Evaluation in iter {num_steps_passed}===================")
                
                calcAcc()
                sttime = time.time()
            num_steps_passed += 1
        pass

# print(gate_results[0][0, 0])
train()
# display_shape()

