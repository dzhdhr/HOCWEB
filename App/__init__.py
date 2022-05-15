from flask import Flask

from App.controllers import init_blueprint
from App.ext import init_ext
from App.setting import envs

'''
Create flask app and calling all the initialization function
'''


def create_app():
    app = Flask(__name__)

    # load in config object
    app.config.from_object(envs.get('develop'))
    # init extensions
    init_ext(app=app)
    init_blueprint(app=app)
    return app
