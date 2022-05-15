import os

import numpy as np
import torch
from flask import Blueprint, request, make_response, send_from_directory
from flask import current_app

from App.Output import status
from App.SimRep import noniterate_detection

from App.numpy_dataloader import NumpyLoader

from App.util import set_device, set_model_pre, init_feature_set, build_T, build_dataset_informal, check_file

detection_controller = Blueprint('detection', __name__)


@detection_controller.route("/here", methods=['GET'])
def test():
    return "detection"


@detection_controller.route("/result", methods=['GET'])
def checkresultnoise():
    print("here")
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


@detection_controller.route("/calculate", methods=['GET'])
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
    output = status(label_file=None, feature_file=f_list[0], use_clip=use_clip, token=token)
    output.from_file(token=token)
    config = {
        'status':output,
        'pre_type': "image",
        "device": set_device(),
        "num_epoch": 1,
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
    sel_noisy, sel_clean, sel_idx = noniterate_detection(config, record, train,output,
                                                         sel_noisy=sel_noisy_rec.copy())
    sel_times_rec[np.array(sel_idx)] += 0.5
    output.noisy_label = np.ndarray.tolist(sel_noisy)
    output.noisy_calculated = True

    output.to_file()
    print(sel_noisy)
    return {"result": sel_noisy.tolist()}
