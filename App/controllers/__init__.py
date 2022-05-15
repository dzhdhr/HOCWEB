from App.controllers.dectection_controller import detection_controller
from App.controllers.estimation_controller import estimation_controller
from App.controllers.log_controller import log_controller
from App.controllers.upload_controller import file_controller


def init_blueprint(app):
    app.register_blueprint(blueprint=detection_controller, url_prefix='/detection')
    # app.register_blueprint(blueprint=hoc_controller, url_prefix="")
    app.register_blueprint(blueprint=log_controller, url_prefix='/log')
    app.register_blueprint(blueprint=estimation_controller, url_prefix='/estimation')
    app.register_blueprint(blueprint=file_controller, url_prefix='/file')