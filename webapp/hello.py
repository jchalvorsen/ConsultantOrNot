from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
from scipy import misc
import numpy as np

from VGG16_gs_model import *


# sett opp nevralt nettverk:
model = VGG_16()
fpath = '../models/vgg16_sg.h5';
model.load_weights(fpath)



# Sett opp webapp
app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'img'
configure_uploads(app, photos)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        # Hent ut prediksjon
        img = misc.imread('./img/' + filename)
        img = misc.imresize(img, (224, 224))
        img = img.transpose()
        img = np.expand_dims(img, axis=0).astype(np.uint8)
        preds, idxs, classes = predict(model, img)     
        return 'You are ' + str(classes[0]) + ' with a confidence of ' + str(int(round(preds[0]*100))) + '%'
    
    return render_template('upload.html')

@app.route("/webcam")
def webcam():
    return render_template('webcam.html')


@app.route("/")
def hello():
    return "Hello Westerdals!"

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5000,ssl_context='adhoc')

