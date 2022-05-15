import numpy as np
import torch

from App.hoc import count_knn_distribution, get_score


def data_transform(record, noise_or_not, sel_noisy):
    # assert noise_or_not is not None
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    noise_or_not_reorder = np.empty(total_len, dtype=bool)
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0
    sel_noisy = np.array(sel_noisy)
    noisy_prior = np.zeros(len(record))

    for item in record:
        for i in item:
            # if i['index'] not in sel_noisy:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            noise_or_not_reorder[cnt] = noise_or_not[i['index']] if noise_or_not is not None else False
            index_rec[cnt] = i['index']
            cnt += 1 - np.sum(sel_noisy == i['index'].item())
            # print(cnt)
        noisy_prior[lb] = cnt - np.sum(noisy_prior)
        lb += 1
    data_set = {'feature': origin_trans[:cnt], 'noisy_label': origin_label[:cnt],
                'noise_or_not': noise_or_not_reorder[:cnt], 'index': index_rec[:cnt]}
    return data_set, noisy_prior / cnt


def get_knn_acc_all_class(args, data_set, output,k=10, noise_prior=None, sel_noisy=None, thre_noise_rate=0.5, thre_true=None):
    # Build Feature Clusters --------------------------------------
    KINDS = args['num_classes']

    all_point_cnt = data_set['feature'].shape[0]
    # global
    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    noise_or_not_sample = data_set['noise_or_not'][sample]
    sel_idx = data_set['index'][sample]
    knn_labels_cnt = count_knn_distribution(args, final_feat, noisy_label, all_point_cnt, k=k, norm='l2')

    method = 'ce'
    # time_score = time.time()
    score = get_score(knn_labels_cnt, noisy_label, k=k, method=method, prior=noise_prior)  # method = ['cores', 'peer']
    # print(f'time for get_score is {time.time()-time_score}')
    score_np = score.cpu().numpy()

    if args['method'] == 'mv':
        # test majority voting
        print(f'Use MV')
        output.noisy_log = output.noisy_log+f'Use MV\n'
        label_pred = np.argmax(knn_labels_cnt, axis=1).reshape(-1)
        sel_noisy += (sel_idx[label_pred != noisy_label]).tolist()
    elif args['method'] == 'rank1':
        print(f'Use rank1')
        Tii_offset = args['Tii_offset']
        print(f'Tii offset is {Tii_offset}')
        f'Tii offset is {Tii_offset}'
        output.noisy_log = output.noisy_log + f'Use rank1'
        # fig=plt.figure(figsize=(15,4))
        for sel_class in range(KINDS):
            thre_noise_rate_per_class = 1 - min(args['Tii_offset'] * thre_noise_rate[sel_class][sel_class], 1.0)
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = (noisy_label.cpu().numpy() == sel_class)
            thre = np.percentile(score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    return sel_noisy


def noniterate_detection(config, record, train_dataset,result, sel_noisy=[]):
    T_given_noisy_true = None
    T_given_noisy = None

    # non-iterate
    # sel_noisy = []
    data_set, noisy_prior = data_transform(record, train_dataset.noise_or_not, sel_noisy)
    # print(data_set['noisy_label'])
    if config['method'] == 'rank1':

        T = np.array(config['status'].T)
        p = np.array(config['status'].p)
        print(T.shape)
        print(p.shape)

        T_given_noisy = T * p / noisy_prior

        print("T given noisy:")
        print(np.round(T_given_noisy, 2))
        result.noisy_log = result.noisy_log+"T given noisy:\n"
        result.noisy_log = result.noisy_log + str(np.round(T_given_noisy, 2))
        # add randomness
        for i in range(T.shape[0]):
            T_given_noisy[i][i] += np.random.uniform(low=-0.05, high=0.05)

    sel_noisy = get_knn_acc_all_class(config, data_set, k=config['k'], noise_prior=noisy_prior, sel_noisy=sel_noisy,
                                      thre_noise_rate=T_given_noisy, thre_true=T_given_noisy_true,output=result)

    sel_noisy = np.array(sel_noisy)
    sel_clean = np.array(list(set(data_set['index'].tolist()) ^ set(sel_noisy)))


    return sel_noisy, sel_clean, data_set['index']
