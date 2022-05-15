import os

from flask import Blueprint, request, make_response, send_from_directory
from flask import current_app
from werkzeug.utils import secure_filename

from App.Output import status
from App.service import get_token

file_controller = Blueprint('file', __name__)


@file_controller.route("/upload", methods=['POST'])
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

    result_path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token, token + ".json")
    result_file = status(token, None, None)

    result_file.from_file(token)
    if type == "feature":
        print("feature")
        result_file.feature_file = secure_filename(f.filename)
    else:
        result_file.label_file = secure_filename(f.filename)
    result_file.to_file()

    return {"token": token}


@file_controller.route("/download/<string:filename>")
def get_result(filename):
    path = os.path.join(os.getcwd(), current_app.config['RESULT_FOLDER'])
    name = filename + ".csv"
    response = make_response(send_from_directory(path, name, as_attachment=True))
    return response


@file_controller.route("/checktoken", methods=['GET'])
def check_token():
    token = request.args.get('token')
    path = os.path.join(os.getcwd(), current_app.config['UPLOAD_FOLDER'], token)
    return {"has_token": os.path.exists(path)}
