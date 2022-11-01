# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import numpy as np
import copy
import torch
from alg.utils import *

class fedstar(torch.nn.Module):
    def __init__(self, args):
        super(fedstar, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args
        self.weight_names_shapes = dict()
        self.para_num = 0
        for name, params in self.server_model.named_parameters():
            self.weight_names_shapes[name] = params.shape
            self.para_num += np.prod(params.shape)
        # server
        self.rec_g_l = 10
        self.rec_g = dict()
        self.thrs_g = args.thrs_g
        for name in self.weight_names_shapes.keys():
            sp = [self.rec_g_l] + list(self.weight_names_shapes[name])
            self.rec_g[name] = torch.zeros(sp).to(self.args.device)
        self.mask_g = dict()
        self.mask_check_g = dict()
        self.freq_g = dict()
        self.length_g = dict()
        for name in self.weight_names_shapes.keys():
            self.mask_g[name] = torch.ones(self.weight_names_shapes[name]).to(self.args.device)
            self.mask_check_g[name] = torch.zeros(self.weight_names_shapes[name]).to(self.args.device)
            self.freq_g[name] = torch.ones(self.weight_names_shapes[name]).to(self.args.device)
            self.length_g[name] = torch.zeros(self.weight_names_shapes[name]).to(self.args.device)
        # client
        self.rec_c_l = 5
        self.rec_c = [dict() for i in range(self.args.n_clients)]
        self.thrs_c = args.thrs_c
        simple_record = dict()
        for name in self.weight_names_shapes.keys():
            sp = [self.rec_c_l] + list(self.weight_names_shapes[name])
            simple_record[name] = torch.zeros(sp).to(self.args.device)
        for i in range(self.args.n_clients):
            self.rec_c[i] = copy.deepcopy(simple_record)

        self.mask_c = [copy.deepcopy(self.mask_g) for i in range(self.args.n_clients)]
        self.mask_check_c = [copy.deepcopy(self.mask_check_g) for i in range(self.args.n_clients)]
        self.freq_c = [copy.deepcopy(self.freq_g) for i in range(self.args.n_clients)]
        self.length_c = [copy.deepcopy(self.length_g) for i in range(self.args.n_clients)]

        # 先放着
        self.mask_s = dict()
        for name in self.weight_names_shapes.keys():
            self.mask_s[name] = torch.ones(self.weight_names_shapes[name]).to(self.args.device)

    def measure_staiblity_server(self):
        for name in self.weight_names_shapes.keys():
            condition = torch.where((self.mask_g[name] == 1) & (self.mask_check_g[name] == 0))
            tmp_mask = measure_stability(self.rec_g[name], self.thrs_g, condition)
            self.mask_g[name][condition] = tmp_mask[condition]

    def measure_staiblity_client(self, client_idx):
        for name in self.weight_names_shapes.keys():
            condition = torch.where((self.mask_c[client_idx][name] == 1) & (self.mask_check_c[client_idx][name] == 0))
            cur_prune = torch.where(self.mask_c[client_idx][name] == 0)
            cur_prune = len(cur_prune[0]) / self.para_num
            if cur_prune < 0.7:
                tmp_mask = measure_stability_c(self.rec_c[client_idx][name], self.thrs_c, condition, 0.7 - cur_prune)
                self.mask_c[client_idx][name][condition] = tmp_mask[condition]

    def client_train(self, c_idx, dataloader, round):
        train_loss, train_acc = trainwithmask(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device,
            self.mask_c[c_idx])
        return train_loss, train_acc

    def update_record_server(self, w_diff):
        update_record(self.rec_g, w_diff, self.rec_g_l)

    def update_record_client(self, c_idx, w_diff):
        update_record(self.rec_c[c_idx], w_diff, self.rec_c_l)

    def server_aggre(self):
        counts = dict()
        client_weights = copy.deepcopy(self.mask_c)
        for name in self.weight_names_shapes.keys():
            counts[name] = torch.zeros_like(self.mask_g[name])
            for c_idx in range(self.args.n_clients):
                counts[name] += self.mask_c[c_idx][name]

            for c_idx in range(self.args.n_clients):
                client_weights[c_idx][name] = torch.where(
                    counts[name] == 0,
                    torch.tensor(0., dtype=torch.float, device=self.args.device),
                    self.mask_c[c_idx][name] / counts[name])

        for key in self.server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.server_model.state_dict()[key].data.copy_ \
                    (self.client_model[0].state_dict()[key])
            else:
                temp = torch.zeros_like(self.server_model.state_dict()[key])
                for c_idx in range(self.args.n_clients):
                    if key in self.weight_names_shapes.keys():
                        temp += self.client_model[c_idx].state_dict()[key] \
                                * client_weights[c_idx][key]
                    else:
                        temp += self.client_model[c_idx].state_dict()[key] \
                                / self.args.n_clients

                if key in self.weight_names_shapes.keys():
                    self.server_model.state_dict()[key].data.copy_(torch.where(
                        (self.mask_g[key] != 0) & (counts[key] != 0),
                        temp,
                        self.server_model.state_dict()[key].data))
                else:
                    self.server_model.state_dict()[key].data.copy_(temp)

    def measure_communication(self, c_idx):
        non_zero_count = 0
        test_zero_count = 0
        for name, params in self.client_model[c_idx].named_parameters():
            condition = torch.where((self.mask_g[name] != 0) & (self.mask_c[c_idx][name] != 0))
            non_zero_count += len(condition[0])

            condition = torch.where((self.mask_c[c_idx][name] != 0))
            test_zero_count += len(condition[0])
        # print(test_zero_count/self.para_num)
        return non_zero_count

    def check_stability_server(self):
        for name in self.weight_names_shapes.keys():
            condition = torch.where(self.mask_g[name] == 0)
            self.length_g[name][condition] += 1
            self.mask_check_g[name] = check_stability(self.mask_check_g[name], self.rec_g[name], self.freq_g[name],
                                                      self.length_g[name], self.thrs_g)
            condition = torch.where(self.mask_check_g[name] == 1)
            self.mask_g[name][condition] = torch.tensor(1., device=self.args.device)

    def check_stability_client(self, c_idx):
        for name in self.weight_names_shapes.keys():
            condition = torch.where(self.mask_c[c_idx][name] == 0)
            self.length_c[c_idx][name][condition] += 1
            self.mask_check_c[c_idx][name] = check_stability(self.mask_check_c[c_idx][name], self.rec_c[c_idx][name],
                                                             self.freq_c[c_idx][name], self.length_c[c_idx][name],
                                                             self.thrs_c)
            condition = torch.where(self.mask_check_c[c_idx][name] == 1)
            self.mask_c[c_idx][name][condition] = torch.tensor(1., device=self.args.device)

    def broadcast_parameter(self):
        for c_idx in range(self.args.n_clients):
            for key in self.server_model.state_dict().keys():
                self.client_model[c_idx].state_dict()[key].data.copy_(
                    self.server_model.state_dict()[key].data)

    def clear_record(self, c_idx):
        for name in self.rec_c[c_idx].keys():
            self.rec_c[c_idx][name] = self.rec_c[c_idx][name] - self.rec_c[c_idx][name]

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

