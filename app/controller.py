import os

import torch
from flask import Blueprint, jsonify, render_template, request, flash

from flask import current_app
from werkzeug.utils import secure_filename

from app.hoc import get_T_P_global
from app.util import set_device, set_model_pre, init_feature_set, build_T, build_dataset_informal

from app.numpy_dataloader import NumpyLoader

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

        train = NumpyLoader(label_path=label_file_name, feature_path=feature_file_name)
        train_dataloader_EF = torch.utils.data.DataLoader(train,
                                                          batch_size=128,
                                                          shuffle=True,
                                                          num_workers=2,
                                                          drop_last=False)

        config = {
            'pre_type': "image",
            "num_classes": train.unique_class(),
            "device": set_device(),
            "dataset": "CIFAR10",
            "label_file_path": ""
        }

        config['P'] = [1.0 / config['num_classes']] * config['num_classes']  # Distribution of 10 clusters
        config['T'] = build_T(config['num_classes'])
        model_pre = set_model_pre(config)
        # flash('start Calculating, please wait')
        config['path'], record, cluster = init_feature_set(config, model_pre, train_dataloader_EF, -1)
        sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, cluster)
        T_est, P_est, T_init, T_err = get_T_P_global(config, sub_noisy_dataset_name, 1501, None, None, lr=0.1)
        T_final = T_est.tolist()
        return render_template('result.html', T=T_final, p=P_est.tolist())
    else:
        return render_template('index.html')
