import enum
import numpy as np
import torch
from network.models import lenet5v
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datautil.prepare_data import get_whole_dataset

def evalandprint_(args, algclass, test_loaders, SAVE_PATH, a_iter):

    acc = np.zeros(args.n_clients)
    for client_idx in range(args.n_clients):
        _, test_acc = algclass.client_eval(
            client_idx, test_loaders[client_idx])
        acc[client_idx] = test_acc
    mean_acc = np.mean(np.array(acc))

    return np.around(mean_acc, 4)

def modelsel(args, device):
    server_model = lenet5v().to(device)
    client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights


def measure_stability(record, threshold, condition):
    if torch.sum(record) == 0:
        mask = torch.ones_like(record[0])
    else:
        sum_1 = torch.zeros_like(record[0])
        sum_2 = torch.zeros_like(record[0])
        for item in record:
            sum_1[condition] += item[condition]
            sum_2[condition] += torch.abs(item[condition])
        stability = torch.where(sum_2[condition] == 0,
                                torch.tensor(0, dtype=torch.float32, device=sum_1.device),
                                torch.abs(sum_1[condition]) / sum_2[condition])
        # 限定

        mask = torch.zeros_like(record[0])
        mask[condition] = torch.where(stability >= threshold, torch.tensor(1., device=sum_1.device),
                                      torch.tensor(0., device=sum_1.device))
    return mask


def measure_stability_c(record, threshold, condition, cur_prune):
    if torch.sum(record) == 0:
        mask = torch.ones_like(record[0])
    else:
        sum_1 = torch.zeros_like(record[0])
        sum_2 = torch.zeros_like(record[0])
        for item in record:
            sum_1[condition] += item[condition]
            sum_2[condition] += torch.abs(item[condition])
        stability = torch.where(sum_2[condition] == 0,
                                torch.tensor(0, dtype=torch.float32, device=sum_1.device),
                                torch.abs(sum_1[condition]) / sum_2[condition])
        if len(stability)!=0:
            tmp = np.percentile(stability.view(-1).cpu().numpy(), cur_prune)
            if threshold < tmp:
                threshold = tmp
        # 限定
        mask = torch.zeros_like(record[0])
        mask[condition] = torch.where(stability >= threshold, torch.tensor(1., device=sum_1.device),
                                      torch.tensor(0., device=sum_1.device))

    return mask


def check_stability(check_mask, record, frequency, length, threashold):
    condition = torch.where(check_mask == 1)
    mask_tmp = measure_stability(record, threashold, condition)
    condition = torch.where((check_mask == 1) & (mask_tmp == 0))
    frequency[condition] = frequency[condition] + 1
    condition = torch.where((check_mask == 1) & (mask_tmp == 1))
    frequency[condition] = (frequency[condition] / 2).int().float()
    frequency = torch.where(frequency < 1, torch.tensor(1., device=frequency.device), frequency)

    check_mask = check_mask - check_mask
    condition = torch.where(frequency <= length)
    check_mask[condition] = check_mask[condition] + 1
    length[condition] = 0
    return check_mask

def update_record(record, w_diff, length):

    for name in w_diff.keys():
        target = torch.zeros_like(w_diff[name])
        condition = torch.where((w_diff[name]!=0)&(record[name][length-1]!=0))
        for i in range(length-1):
            record[name][i][condition] = record[name][i+1][condition]
        record[name][length-1][condition] = w_diff[name][condition]
        w_diff[name][condition] = w_diff[name][condition] - w_diff[name][condition]

        if w_diff[name].equal(target)==False:
            for i in range(length-1, 0, -1):
                condition = torch.where((w_diff[name]!=0)&
                                        (record[name][i]==0)&(record[name][i-1]!=0))
                record[name][i][condition] = w_diff[name][condition]
                w_diff[name][condition] = w_diff[name][condition]-w_diff[name][condition]

        if w_diff[name].equal(target) == False:
            condition = torch.where((w_diff[name]!=0)&(record[name][0]==0))
            record[name][0][condition] = w_diff[name][condition]


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total

def trainwithmask(model, data_loader, optimizer, loss_fun, device, mask):

    model.train()
    loss_all = 0
    total = 0
    correct = 0

    for data, target in data_loader:

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()

        for name, params in model.named_parameters():
            if 'weight' in name:
                params.grad.data.copy_(params.grad.data * mask[name].to(device))

        optimizer.step()

    return loss_all / len(data_loader), correct/total
