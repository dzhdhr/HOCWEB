from flask import Flask

from app.controller import init_blueprint
from app.ext import init_ext
from app.setting import envs

'''
Create flask app and calling all the initialization function
'''


def create_app():
    app = Flask(__name__)

    # load in config object
    app.config.from_object(envs.get('develop'))
    # init extensions
    init_ext(app=app)
    # init blueprint
    init_blueprint(app=app)

    return app
