# Classifier API endpoints

from base64 import b64decode
from PIL import Image
from io import BytesIO
from classifier.network import *

def query(body):
    global neuralnet

    header, encoded = body.decode("utf-8").split(",", 1)
    fileformat = header[5:] # Remove the ':data' part
    fileformat = fileformat.split(';')[0] # Strip it to the bare http file type (image/png)
    extension = "." + fileformat.split('/')[1] # Set the local file extension (.png)

    im = Image.open(BytesIO(b64decode(encoded)))
    resize(im, 64)

    return make_prediction.predict(neuralnet, im)
