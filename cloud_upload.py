import os
import pyrebase
from APIKeys import configure

def cloud_upload(path):
    conf = configure

    firebase = pyrebase.initialize_app(conf)
    storage = firebase.storage()
    storage.child(path).put(path)

    os.remove(path)