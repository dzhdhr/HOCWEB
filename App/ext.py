from flask_bootstrap import Bootstrap
from flask_cors import CORS


def init_ext(app):
    CORS(app)
    Bootstrap(app)
