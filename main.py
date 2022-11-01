# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import os
import numpy as np
import torch
import argparse
import numpy as np
import random

from alg.starfl import fedstar
from alg.utils import evalandprint_
from datautil.prepare_data import *
from alg import starfl
from alg import utils

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_config(parser):
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='[medmnist, medmnistA]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./result/', help='path to save the result')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients')
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--datapercent', type=float,
                        default=1e-1, help='data percent to use')
    # algorithm-specific parameters
    parser.add_argument('--thrs_g', type=float, default=0, help='threshold to server stability')
    parser.add_argument('--thrs_c', type=float, default=0, help='threshold to client stability')
    args = parser.parse_args()
    args.random_state = np.random.RandomState(5)
    set_random_seed(args.seed)

    return args

def create_save_path(args):
    exp_folder = f'fed_{args.dataset}_{args.non_iid_alpha}_{args.iters}_{args.wk_iters}_{args.thrs_c}_{args.thrs_g}'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path)
    return SAVE_PATH


if __name__ == '__main__':

    # 设置配置参数
    parser = argparse.ArgumentParser()
    args = read_config(parser)
    SAVE_PATH = create_save_path(args)

    # 构建数据集
    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)

    algclass = fedstar(args=args)

    rec_acc = []
    a_iter = 0

    while(a_iter<=args.iters):

        print(f"============ Train round {a_iter} ============")

        # measure the stability in the server
        algclass.measure_staiblity_server()

        # broadcast the global parameters
        algclass.broadcast_parameter()

        # main training
        for client_idx in range(args.n_clients):

            # measure the stability in the client
            algclass.measure_staiblity_client(client_idx)

            # clear record
            algclass.clear_record(client_idx)

            for wi in range(args.wk_iters):
                # record former
                if wi <= algclass.rec_c_l:
                    client_weight = dict()
                    for name, params in algclass.client_model[client_idx].named_parameters():
                        client_weight[name] = copy.deepcopy(params.data)

                # train with the mask
                algclass.client_train(client_idx, train_loaders[client_idx], a_iter)

                # record update
                if wi <= algclass.rec_c_l:
                    for name, params in algclass.client_model[client_idx].named_parameters():
                        client_weight[name] -= copy.deepcopy(params.data)
                    algclass.update_record_client(client_idx, client_weight)

            # check the stability of the client
            algclass.check_stability_client(client_idx)

        # evaluate the personalize model
        mean_acc = evalandprint_(args, algclass, test_loaders, SAVE_PATH, a_iter)
        rec_acc.append(mean_acc)

        # record the former
        server_weight = dict()
        for name, params in algclass.server_model.named_parameters():
            server_weight[name] = copy.deepcopy(params.data)

        # aggregate the model
        algclass.server_aggre()
        # recorde update
        for name, params in algclass.server_model.named_parameters():
            server_weight[name] -= params.data
        algclass.update_record_server(server_weight)

        # check the stability of the server
        algclass.check_stability_server()

        a_iter += 1

        print('Personalized test acc for each client: ')
        if a_iter < 50:
            print(rec_acc)
        else:
            print(max(rec_acc))

    np.save(SAVE_PATH+'_rec_acc.npy', rec_acc)

