class Config:
    DEBUG = False
    TESTING = False


class DevelopConfig(Config):
    DEBUG = True
    UPLOAD_FOLDER = 'upload/'
    secret_key = 'super secret key'
    RESULT_FOLDER = 'result/'
    DETECT_RESULT = 'detectreuslt/'
    DETECT_LOG='detectlog/'


envs = {
    'develop': DevelopConfig,
}
