import os

import torch
from flask import Blueprint, jsonify, render_template, request, flash

from flask import current_app
from werkzeug.utils import secure_filename

from App.hoc import get_T_P_global
from App.util import set_device, set_model_pre, init_feature_set, build_T, build_dataset_informal

from App.numpy_dataloader import NumpyLoader

import numpy as np

hoc_controller = Blueprint('hoc', __name__)


def init_blueprint(app):
    app.register_blueprint(blueprint=hoc_controller, url_prefix="")


@hoc_controller.route("/", methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        feature_file = request.files['feature-file']
        label_file = request.files['label-file']

        label_file_name = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(label_file.filename))
        label_file.save(label_file_name)

        feature_file_name = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(feature_file.filename))
        feature_file.save(feature_file_name)



        config = {
            'pre_type': "image",
            "device": set_device(),
            "dataset": "CIFAR10",
            "label_file_path": "",
            "num_epoch":1
        }
        model_pre, pre_process = set_model_pre(config)
        train = NumpyLoader(label_path=label_file_name, feature_path=feature_file_name, pre_process=pre_process)
        config['num_classes'] = train.unique_class()
        config['P'] = [1.0 / config['num_classes']] * config['num_classes']  # Distribution of 10 clusters
        config['T'] = build_T(config['num_classes'])

        train_dataloader_EF = torch.utils.data.DataLoader(train,
                                                          batch_size=128,
                                                          shuffle=True,
                                                          num_workers=2,
                                                          drop_last=False)
        # flash('start Calculating, please wait')
        logger = open("./log/test", 'w')
        logger.write("Extracting Feature\n")
        logger.flush()
        config['path'], record, cluster = init_feature_set(config, model_pre, train_dataloader_EF, -1)
        sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, cluster)
        logger.write("Extracting Feature --- Done\n")
        logger.write("\n")
        logger.write("Estimating noise transition matrix T and clean prior p using HOC Global\n")
        logger.flush()

        T_est, P_est, T_init = get_T_P_global(config, sub_noisy_dataset_name,logger, 1501, None, None, lr=0.1)
        T_est = np.around(T_est*100, decimals = 1)
        P_est = np.around(P_est*100, decimals = 1)
        T_final = T_est.tolist()
        logger.seek(0)
        logger.truncate()
        logger.close()
        torch.cuda.empty_cache()
        return render_template('result.html', T=T_final, p=P_est.tolist())
    else:
        return render_template('index.html')


@hoc_controller.route("/getlog")
def get_log():
    f = open("./log/test")
    log = f.read()
    f.close()
    return log
