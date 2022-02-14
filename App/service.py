import uuid


def get_token():
    name = str(uuid.uuid4())
    return name
