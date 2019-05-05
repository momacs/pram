import os

class DevConfig():
    SECRET_KEY = os.urandom(24)
    SESSION_TYPE = 'redis'
    # SESSION_REDIS = 'redis://localhost:6379/1'
    SESSION_COOKIE_NAME = 'pram-dev'
    # SESSION_PERMANENT = True
    # PERMANENT_SESSION_LIFETIME = timedelta(days=31) (2678400 seconds)
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

class ProdConfig():
    SECRET_KEY = os.urandom(24)
    SESSION_TYPE = 'redis'
    SECRET_KEY = os.urandom(24)
    SESSION_COOKIE_NAME = 'pram'
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
