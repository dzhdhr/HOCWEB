from flask_script import Manager

from App import create_app

app = create_app()
manager = Manager(app=app)

# app entry point
if __name__ == '__main__':
    manager.run()
