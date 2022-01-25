import time

import torch.nn.functional as F

from App.util import *


def distCosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - cosine distance
    return dist


def count_y(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    max_val = np.max(dist)
    am = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist, axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt


def func(KINDS, p_estimate, T_out, P_out, N, step, LOCAL, _device):
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)  # define the loss

    P = smp(P_out)
    T = smt(T_out)

    mode = random.randint(0, KINDS - 1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N, N*N*N
    p_temp = count_real(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    weight = [1.0, 1.0, 1.0]
    # weight = [2.0,1.0,1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j])  # / np.sqrt(N**j)

    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P + eps)) / 10

    return loss


def calc_func(KINDS, p_estimate, LOCAL, _device, logger, max_step=501, T0=None, p0=None, lr=0.1):
    # init
    # _device =  torch.device("cpu")
    N = KINDS
    eps = 1e-8
    if T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N, N))
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device=None) / N + torch.rand((N, 1), device=None) * 0.1  # P：0-9 distribution
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr=lr)

    # train
    loss_min = 100.0
    T_rec = torch.zeros_like(T)
    P_rec = torch.zeros_like(P)

    time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(KINDS, p_estimate, T, P, N, step, LOCAL, _device)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        if step % 100 == 0:
            print('loss {}'.format(loss))
            logger.write('loss {}\n'.format(loss))
            print(f'step: {step}  time_cost: {time.time() - time1}')
            logger.write(f'step: {step}/{max_step - 1}  time_cost: {time.time() - time1}\n')
            print(f'T {np.round(smt(T.cpu()).detach().numpy() * 100, 1)}', flush=True)
            logger.write(f'T {np.round(smt(T.cpu()).detach().numpy() * 100, 1)}\n')
            print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy() * 100, 1)}', flush=True)
            logger.write(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy() * 100, 1)}\n')
            logger.flush()
            time1 = time.time()

    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def get_T_P_global(config, sub_noisy_dataset_name, logger, max_step=501, T0=None, p0=None, lr=0.1, ):
    global GLOBAL_T_REAL
    KINDS = config['num_classes']
    data_set = torch.load(f'{sub_noisy_dataset_name}', map_location=torch.device('cpu'))

    # all_point_cnt = 5000

    # all_point_cnt = 5000 if data_set['feature'].shape[0]>5000 else data_set['feature'].shape[0]
    all_point_cnt = max(data_set['feature'].shape[0] // 2, KINDS ** 2 * 2)
    # all_point_cnt = 2000

    # NumTest = int(max(20,data_set['feature'].shape[0]//5000)) if all_point_cnt == 5000 else 1
    NumTest = int(20)
    # TODO: make the above parameters configurable

    print(f'Estimating global T. Sampling {all_point_cnt} examples each time')

    logger.write(f'Estimating high-order consensuses (numerically). Sampling {all_point_cnt} examples each time\n')
    logger.flush()

    # Build Feature Clusters --------------------------------------
    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)

    for idx in range(NumTest):
        print(idx, flush=True)
        logger.write(f"{idx + 1}/{NumTest}\n")
        logger.flush()
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
            # ss = torch.abs(p_estimate[i] / (idx + 1) - p_real[i])
            # p_estimate_rec[idx, i] = torch.mean(torch.abs(p_estimate[i] / (idx + 1) - p_real[i])) * 100.0 / (
            #     torch.mean(p_real[i]))  # Assess the gap between estimation value and real value
        # print(p_estimate_rec[idx], flush=True)
        # logger.writelines(str(p_estimate_rec[idx])+"\n")

    logger.write(f'Estimating high-order consensuses (numerically) --- Done\n')
    logger.flush()
    logger.write(f'\n')
    logger.write(f'Solving equations:\n')

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, config['device'], logger, max_step, T0, p0,
                                                 lr=lr)
    P_calc = P_calc.view(-1).cpu().numpy()
    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()

    # print("----Real value----------")
    # print(f'Real: P = {P_real},\n T = \n{np.round(np.array(T_real),3)}')
    # print(f'Sum P = {sum(P_real)},\n sum T = \n{np.sum(np.array(T_real), 1)}')
    # print("\n----Calc result----")
    # print(f"loss = {loss_min}, \np = {P_calc}, \nT_est = \n{np.round(E_calc, 3)}")
    # print(f"sum p = {np.sum(P_calc)}, \nsum T_est = \n{np.sum(E_calc, 1)}")
    # print("\n---Error of the estimated T (sum|T_est - T|/N * 100)----", flush=True)
    # print(f"L11 Error (Global): {np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / KINDS * 100}")
    # logger.write(f"L11 Error (Global): {np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / KINDS * 100}\n")
    # T_err = np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / KINDS * 100
    # rec_global = [[] for _ in range(3)]
    # rec_global[0], rec_global[1], rec_global[2] = loss_min, T_real, E_calc
    # path = "./rec_global/" + config['dataset'] + "_" + config['label_file_path'][11:14] + "_" + config[
    #     'pre_type'] + ".pt"
    # torch.save(rec_global, path)
    logger.flush()
    return E_calc, P_calc, T_init


def count_knn_distribution(args, feat_cord, label, cluster_sum, k, norm='l2'):
    # feat_cord = torch.tensor(final_feat)
    KINDS = args['num_classes']
    dist = cosDistance(feat_cord)

    print(f'knn parameter is k = {k}')
    time1 = time.time()
    min_similarity = args['min_similarity']
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]
    knn_labels = label[indices]

    knn_labels_cnt = torch.zeros(cluster_sum, KINDS)

    for i in range(KINDS):
        knn_labels_cnt[:, i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i), 1)

    time2 = time.time()
    print(f'Running time for k = {k} is {time2 - time1}')

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob


def get_score(knn_labels_cnt, label, k, method='cores', prior=None):  # method = ['cores', 'peer']
    # knn_labels_cnt: sampleSize * #class
    # knn_labels_cnt /= (k*1.0)
    # import pdb
    # pdb.set_trace()
    loss = F.nll_loss(torch.log(knn_labels_cnt + 1e-8), label, reduction='none')
    # loss = -torch.tanh(-F.nll_loss(knn_labels_cnt, label, reduction = 'none')) # TV
    # loss = -(-F.nll_loss(knn_labels_cnt, label, reduction = 'none')) #
    # loss_numpy = loss.data.cpu().numpy()
    # num_batch = len(loss_numpy)
    # loss_v = np.zeros(num_batch)
    # loss_div_numpy = float(np.array(0))

    # loss_ = -(knn_labels_cnt)   #
    # loss_ = -torch.tanh(knn_labels_cnt)   # TV
    # import pdb
    # pdb.set_trace()
    loss_ = -torch.log(knn_labels_cnt + 1e-8)
    if method == 'cores':
        score = loss - torch.mean(loss_, 1)
        # score =  loss
    elif method == 'peer':
        prior = torch.tensor(prior)
        score = loss - torch.sum(torch.mul(prior, loss_), 1)
    elif method == 'ce':
        score = loss
    elif method == 'avg':
        score = - torch.mean(loss_, 1)
    elif method == 'new':
        score = 1.1 * loss - torch.mean(loss_, 1)
    else:
        raise NameError('Undefined method')

    return score
