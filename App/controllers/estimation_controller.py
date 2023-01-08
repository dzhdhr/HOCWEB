import json

from flask import Blueprint
import os

import numpy as np
import torch
from flask import Blueprint, request, make_response, send_from_directory
from flask import current_app

from App.Output import status
from App.hoc import get_T_P_global
from App.numpy_dataloader import NumpyLoader
from App.util import set_device, set_model_pre, init_feature_set, build_T, build_dataset_informal, check_file

estimation_controller = Blueprint('estimate', __name__)


@estimation_controller.route("/calculate", methods=['GET'])
def calculate():
    #get param
    token = request.args.get('token')
    step = 1200
    use_clip = False

    output = status(token,None,None,step,use_clip)
    output.from_file(token)
    use_clip = output.use_clip
    print(use_clip)
    output.step = step
    output.use_clip = use_clip
    result_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, token + ".json")
    output.to_file()

    label_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label',output.label_file)
    feature_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature', output.feature_file)
    config = {
        'pre_type': "image",
        "device": set_device(),
        "dataset": "CIFAR10",
        "label_file_path": "",
        "num_epoch": 1
    }
    model_pre, pre_process = set_model_pre(config)
    if not use_clip:
        pre_process = None
    train = NumpyLoader(label_path=label_path, feature_path=feature_path, pre_process=pre_process)
    config['num_classes'] = train.unique_class()
    config['P'] = [1.0 / config['num_classes']] * config['num_classes']  # Distribution of 10 clusters
    config['T'] = build_T(config['num_classes'])

    train_dataloader_EF = torch.utils.data.DataLoader(train,
                                                      batch_size=64,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
    logger = open("./log/" + token, 'a')
    output.matrix_log = output.matrix_log+"Extracting Feature\n"
    output.to_file()
    logger.write("Extracting Feature\n")
    logger.flush()

    config['path'], record, cluster = init_feature_set(config, model_pre, train_dataloader_EF, -1,output, use_clip)
    sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, cluster)
    logger.write("Extracting Feature --- Done\n")
    logger.write("\n")
    logger.write("Estimating noise transition matrix T and clean prior p using HOC Global\n")
    output.matrix_log = output.matrix_log+"Estimating noise transition matrix T and clean prior p using HOC Global\n"
    logger.flush()
    output.to_file()
    T_est, P_est, T_init = get_T_P_global(config, sub_noisy_dataset_name, logger, step, None, None, lr=0.1,status=output)
    # output.to_file(result_path)
    T_est = np.round(T_est, decimals=4)
    P_est = np.round(P_est, decimals=4)

    logger.close()

    torch.cuda.empty_cache()

    output.T = np.ndarray.tolist(T_est)
    output.matrix_calculated = True
    output.p = np.ndarray.tolist(P_est)

    with open(result_path, "w") as outfile:
        json.dump(output.to_json(), outfile)

    # np.savetxt("./result/" + token + "_T.csv", T_est, delimiter=",")
    # np.savetxt("./result/" + token + "_P.csv", P_est, delimiter=',')
    return {'T': np.ndarray.tolist(T_est), 'P': np.ndarray.tolist(P_est)}


@estimation_controller.route("", methods=['GET'])
def checkstatus():
    token = request.args.get('token')
    result = status(token,None,None)
    result.from_file(token=token)
    return result.to_json()

    # result_path = os.path.join(os.getcwd(), current_app.config['RESULT_FOLDER'], token + '_T.csv')
    # has_result = os.path.exists(result_path)
    # log_path = os.path.join(os.getcwd(), 'log', token)
    # has_log = os.path.exists(log_path)
    #
    # feature_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature')
    #
    # label_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label')
    # if not has_result:
    #     return {'calculated': has_result, 'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path),
    #             'calculating': has_log}
    # T = np.ndarray.tolist(np.round(np.genfromtxt(result_path, delimiter=','), 2))
    # return {'calculated': has_result, 'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path),
    #         'calculating': has_log, "payload": T}
