from flask import Blueprint, request

from App.Output import status

log_controller = Blueprint('log', __name__)


@log_controller.route("/test", methods=['GET'])
def test():
    return "log"


@log_controller.route("")
def get_log():
    token = request.args.get('token')
    result_file = status(token, None, None)
    result_file.from_file(token)
    return {"log": result_file.matrix_log,
            "batchProgress": 100 * result_file.current_batch / result_file.total_batch,
            "stepProgress": 100 * result_file.current_step / result_file.step,
            }


@log_controller.route("/noise")
def get_noisy_log():
    token = request.args.get('token')
    result_file = status(token, None, None)
    result_file.from_file(token)
    return {"log": result_file.noisy_log}
