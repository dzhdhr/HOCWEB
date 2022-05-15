import json
import os.path

from flask import current_app


class status:

    def __init__(self, token, feature_file, label_file, step=1500, use_clip=False, ):
        self.token = token
        self.step = step
        self.use_clip = use_clip
        self.feature_file = feature_file
        self.label_file = label_file
        self.matrix_calculated = False
        self.has_matrix_log = False
        self.T = None
        self.noisy_calculated = False
        self.noisy_label = None
        self.matrix_log = ""
        self.noisy_log = ""
        self.p = None
        self.current_step = 0
        self.total_batch = 1
        self.current_batch = 0

    def from_file(self, token):
        self.token = token
        status_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, token + ".json")
        if not os.path.exists(status_path):
            return
        with open(status_path) as json_file:
            data = json.load(json_file)
            self.step = data['step']
            self.use_clip = data['clip']
            self.feature_file = data['featureFile']
            self.label_file = data['labelFile']
            self.matrix_calculated = data['matrixCalculated']
            self.has_matrix_log = data['hasMatrixLog']
            self.T = data['T']
            self.noisy_calculated = data['noisyLabelCalculated']
            self.noisy_label = data['noisyLabel']
            self.matrix_log = data['matrixLog']
            self.current_step = data['currentStep']
            self.current_batch = data['currentBatch']
            self.total_batch = data['totalBatch']
            self.noisy_log = data['noisyLog']
            self.p = data['p']

    def to_json(self):
        return {
            "token": self.token,
            "featureFile": self.feature_file,
            "labelFile": self.label_file,
            "step": self.step,
            "clip": self.use_clip,
            "matrixCalculated": self.matrix_calculated,
            "hasMatrixLog": self.has_matrix_log,
            "noisyLog": self.noisy_log,
            "matrixLog": self.matrix_log,
            "T": self.T,
            "noisyLabelCalculated": self.noisy_calculated,
            "noisyLabel": self.noisy_label,
            "p": self.p,
            "totalBatch": self.total_batch,
            "currentBatch":self.current_batch,
            "currentStep": self.current_step
        }

    def to_file(self):
        path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], self.token, self.token + ".json")
        with open(path, "w") as outfile:
            json.dump(self.to_json(), outfile)
