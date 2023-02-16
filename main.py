from flask import Flask, request, jsonify
from PIL import Image
from classify.predict import UseModel

import os
from preprocess import read_and_process
import time

app = Flask(__name__)

model = UseModel(weights=["car_reco_model.pt"])

@app.route("/predict", methods=["POST"])
def predict():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        path = f"./to_predict/{int(time.time() * 1000000)}.jpg"
        read_and_process(im_bytes, path)
        result = model.predict(path)
        os.remove(path) # clear storage after finish process

    return jsonify({'msg': 'success', 'predicted': result[0]})

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

        