import os
import uuid

import numpy as np
import torch
from flask import Blueprint, request, make_response, send_from_directory
from flask import current_app
from werkzeug.utils import secure_filename

from App.SimRep import noniterate_detection
from App.hoc import get_T_P_global
from App.numpy_dataloader import NumpyLoader
from App.service import get_token
from App.util import set_device, set_model_pre, init_feature_set, build_T, build_dataset_informal, check_file

hoc_controller = Blueprint('hoc', __name__)


def init_blueprint(app):
    app.register_blueprint(blueprint=hoc_controller, url_prefix="")


@hoc_controller.route("/getlog")
def get_log():
    f = open("./log/" + request.args.get('file'))
    log = f.read()
    f.close()
    return {"log": log}


@hoc_controller.route("/getresult/<string:filename>")
def get_result(filename):
    path = os.path.join(os.getcwd(), current_app.config['RESULT_FOLDER'])
    name = filename + ".csv"
    response = make_response(send_from_directory(path, name, as_attachment=True))
    return response


@hoc_controller.route("/upload", methods=['POST'])
def uploadfile():
    token = request.form.get('token')
    print(token)
    if token is None:
        print("gen new token")
        token = get_token()
    path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token)
    print(token)

    if not os.path.exists(path):
        os.mkdir(path)
    f = request.files['file']
    type = request.form.get("fileType")
    type_dir = os.path.join(path, type)
    if not os.path.exists(type_dir):
        os.mkdir(type_dir)
    file_path = os.path.join(path, type, secure_filename(f.filename))
    f.save(file_path)
    return {"token": token}


@hoc_controller.route("/checkstatus")
def checkstatus():
    token = request.args.get('token')
    result_path = os.path.join(os.getcwd(), current_app.config['RESULT_FOLDER'], token + '_T.csv')
    has_result = os.path.exists(result_path)
    log_path = os.path.join(os.getcwd(), 'log', token)
    has_log = os.path.exists(log_path)

    feature_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature')

    label_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label')
    if not has_result:
        return {'calculated': has_result, 'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path),
                'calculating': has_log}
    T = np.ndarray.tolist(np.round(np.genfromtxt(result_path, delimiter=','), 2))
    return {'calculated': has_result, 'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path),
            'calculating': has_log, "payload": T}


@hoc_controller.route("/checkdetectionresult")
def checkresultnoise():
    token = request.args.get('token')
    result_path = os.path.join(os.getcwd(), current_app.config['DETECT_RESULT'], token + '_result.csv')
    has_result = os.path.exists(result_path)
    log_path = os.path.join(os.getcwd(), current_app.config['DETECT_LOG'], token)
    has_log = os.path.exists(log_path)
    feature_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature')
    label_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label')
    matrix_result_path = os.path.join(os.getcwd(), current_app.config['RESULT_FOLDER'], token + '_T.csv')
    has_matrix = os.path.exists(matrix_result_path)
    if not has_result:
        return {'calculated': has_result, 'calculating': has_log, "matrix": has_matrix,
                'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path)}
    result = np.ndarray.tolist(np.genfromtxt(result_path, delimiter=','))
    return {'calculated': has_result, 'feature': os.path.exists(feature_path), 'label': os.path.exists(label_path),
            'calculating': has_log, "payload": result, "matrix": has_matrix}


@hoc_controller.route("/calculate")
def calculate():
    token = request.args.get('token')
    step = 1500
    use_clip = False
    temp_path_1 = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature')
    f_list = os.listdir(temp_path_1)
    feature_path = os.path.join(temp_path_1, f_list[0])

    temp_path_2 = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label')
    f_list_2 = os.listdir(temp_path_2)
    label_path = os.path.join(temp_path_2, f_list_2[0])
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
    logger.write("Extracting Feature\n")
    logger.flush()

    config['path'], record, cluster = init_feature_set(config, model_pre, train_dataloader_EF, -1, use_clip)
    sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, cluster)
    logger.write("Extracting Feature --- Done\n")
    logger.write("\n")
    logger.write("Estimating noise transition matrix T and clean prior p using HOC Global\n")
    logger.flush()

    T_est, P_est, T_init = get_T_P_global(config, sub_noisy_dataset_name, logger, step, None, None, lr=0.1)
    T_est = np.round(T_est, decimals=4)
    P_est = np.round(P_est, decimals=4)

    logger.seek(0)
    logger.truncate()
    logger.close()

    torch.cuda.empty_cache()

    np.savetxt("./result/" + token + "_T.csv", T_est, delimiter=",")
    np.savetxt("./result/" + token + "_P.csv", P_est, delimiter=',')
    return {'T': np.ndarray.tolist(T_est), 'P': np.ndarray.tolist(P_est)}


@hoc_controller.route("/getnoise")
def calculate_noise():
    sel_noisy_rec = []
    token = request.args.get('token')

    use_clip = False
    temp_path_1 = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'feature')
    f_list = os.listdir(temp_path_1)
    feature_path = os.path.join(temp_path_1, f_list[0])

    temp_path_2 = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, 'label')
    f_list_2 = os.listdir(temp_path_2)
    label_path = os.path.join(temp_path_2, f_list_2[0])

    config = {
        'pre_type': "image",
        "device": set_device(),
        "num_epoch": 1,
        "T_path": f"./result/{token}_T.csv",
        "P_path": f"./result/{token}_P.csv",
        "k": 10,
        "min_similarity": 0.0,
        "Tii_offset": 1.0
    }
    model_pre, pre_process = set_model_pre(config)

    if not use_clip:
        pre_process = None
    train = NumpyLoader(label_path=label_path, feature_path=feature_path, pre_process=pre_process)
    config['num_classes'] = train.unique_class()
    config['num_training_samples'] = train.label.shape[0]
    train_dataloader_EF = torch.utils.data.DataLoader(train,
                                                      batch_size=256,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      drop_last=False)
    model_pre.eval()
    config['size'] = train.label.shape[0]
    sel_clean_rec = np.zeros((config['num_epoch'], config['size']))
    sel_times_rec = np.zeros(config['size'])
    record = [[] for _ in range(config['num_classes'])]
    for i_batch, (feature, label, index) in enumerate(train_dataloader_EF):
        feature = feature.to(config['device'])
        label = label.to(config['device'])
        extracted_feature = feature.reshape(feature.shape[0], -1)
        for i in range(extracted_feature.shape[0]):
            record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})
    config['method'] = 'rank1'
    sel_noisy, sel_clean, sel_idx = noniterate_detection(config, record, train,
                                                         sel_noisy=sel_noisy_rec.copy())
    sel_times_rec[np.array(sel_idx)] += 0.5

    result_path = os.path.join(os.getcwd(), current_app.config['DETECT_RESULT'],
                               token + '_result.csv')
    np.savetxt(result_path, sel_noisy, delimiter=',')
    print(sel_noisy)
    return {"result": sel_noisy.tolist()}


@hoc_controller.route("/checktoken", methods=['GET'])
def check_token():
    token = request.args.get('token')
    path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token)
    return {"has_token": os.path.exists(path)}
