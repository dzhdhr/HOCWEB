class Config:
    DEBUG = False
    TESTING = False


class DevelopConfig(Config):
    DEBUG = True
    UPLOAD_FOLDER = 'upload/'
    secret_key = 'super secret key'


envs = {
    'develop': DevelopConfig,
}
