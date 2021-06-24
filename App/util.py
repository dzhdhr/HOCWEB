import gzip

import numpy as np
import torch
import random
import App.resnet_image as res_image
import torch.nn as nn

smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)


def unzip(file_path):
    file_name = file_path.replace('.gz', '')

    g_file = gzip.GzipFile(file_name)
    open(file_name, "wb+").write(g_file.read())
    g_file.close()


def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device


def set_model_pre(config):
    # use resnet50 for ImageNet pretrain (PyTorch official pre-trained model)
    if config['pre_type'] == 'image':
        model = res_image.resnet50(pretrained=True)
    else:
        RuntimeError('Undefined pretrained model.')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])
    model.to(config['device'])
    return model


def init_feature_set(config, model_pre, train_dataloader, rnd):
    c1m_cluster_each = [0 for _ in range(config['num_classes'])]
    # save the 512-dim feature as a dataset
    model_pre.eval()
    record = [[] for _ in range(config['num_classes'])]

    for i_batch, (feature, label, index) in enumerate(train_dataloader):
        feature = feature.to(config['device'])
        label = label.to(config['device'])
        extracted_feature, _ = model_pre(feature)
        for i in range(extracted_feature.shape[0]):
            record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})

    path = f'./data/{config["pre_type"]}_{config["label_file_path"][7:-3]}.pt'
    return path, record, c1m_cluster_each


def extract_sub_dataset(sub_cluster_each, origin, sub_clean_dataset_name, sub_noisy_dataset_name=None):
    for i in range(len(sub_cluster_each)):  # KINDS
        random.shuffle(origin[i])
        origin[i] = origin[i][:sub_cluster_each[i]]

        for ori in origin[i]:
            ori['label'] = i

    total_len = sum([len(a) for a in origin])

    origin_trans = torch.zeros(total_len, origin[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    origin_index = torch.zeros(total_len).long()
    cnt = 0
    for item in origin:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = i['label']
            origin_index[cnt] = i['index']
            cnt += 1
    torch.save({'feature': origin_trans, 'clean_label': origin_label, 'index': origin_index},
               f'{sub_clean_dataset_name}')
    origin_dataset = torch.load(f'{sub_clean_dataset_name}')
    origin_dataset['noisy_label'] = origin_dataset['clean_label'].clone()
    torch.save(origin_dataset, f'{sub_noisy_dataset_name}')


def build_dataset_informal(config, data, c1m_cluster_each):
    sub_clean_dataset_name = f'{config["path"][:-3]}_clean.pt'
    sub_noisy_dataset_name = f'{config["path"][:-3]}_noisy.pt'
    sub_cluster_each = [int(50000 / config['num_classes'])] * config["num_classes"]
    extract_sub_dataset(sub_cluster_each, data, sub_clean_dataset_name,
                        sub_noisy_dataset_name)

    return sub_clean_dataset_name, sub_noisy_dataset_name


def check_T_torch(KINDS, clean_label, noisy_label):
    T_real = np.zeros((KINDS, KINDS))
    for i in range(clean_label.shape[0]):
        T_real[clean_label[i]][noisy_label[i]] += 1
    P_real = [sum(T_real[i]) * 1.0 for i in range(KINDS)]  # random selection
    for i in range(KINDS):
        if P_real[i] > 0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real) / sum(P_real)
    print(f'Check: P = {P_real},\n T = \n{np.round(T_real, 3)}')
    return T_real, P_real


def count_real(KINDS, T, P, mode, _device='cpu'):
    # time1 = time.time()
    P = P.reshape((KINDS, 1))
    p_real = [[] for _ in range(3)]

    p_real[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    # p_real[2] = torch.zeros((KINDS, KINDS, KINDS)).to(_device)
    p_real[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        p_real[1] = torch.cat([p_real[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with p_estimate
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                p_real[2][r][(i + r + KINDS) % KINDS] = temp33[r]
        else:
            p_real[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]

    temp = []  # adjust the order of the output (N*N), keeping consistent with p_estimate
    for p1 in range(KINDS):
        temp = torch.cat((p_real[1][p1, KINDS - p1:], p_real[1][p1, :KINDS - p1]))
        p_real[1][p1] = temp
    return p_real


def build_T(cluster):
    T = [[0 for _ in range(cluster)] for _ in range(cluster)]
    for i in range(cluster):
        rand_sum = 0
        for j in range(cluster):
            if i != j:
                rand = round(random.uniform(0.01, 0.07), 3)
                rand_sum += rand
                T[i][j] = rand
        T[i][i] = 1 - rand_sum
    return T
