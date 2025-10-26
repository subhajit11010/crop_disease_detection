from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64, io
import numpy as np
from inference_sdk import InferenceHTTPClient


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

crop_type = None
img_arr = None
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="QFsh3LL5uFtkMHKrLf7V"
)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def classify_crop():
    global crop_type, img_arr
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No Image Found"})
    
    crop_type = data["crop_type"]
    image_data = data["image"].split(',')[-1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_arr = np.array(image)
    return predict_disease(img_arr, crop_type)
    

def predict_disease(img_arr, crop_type):
    if crop_type == "rice":
        response = CLIENT.infer(img_arr, model_id="rice-leaf-disease-detection-obj/1")
    elif crop_type == "wheat":
        response = CLIENT.infer(img_arr, model_id="wh-gfqvv/1")
    elif crop_type == "tomato":
        response = CLIENT.infer(img_arr, model_id="tomato-leaf-disease-rxcft/3")
    
    if response and response.get("predictions"):
        predictions = []
        for prd in response["predictions"]:
            single_pred = prd.get("class")
            if single_pred not in predictions:
                predictions.append(single_pred)
        return jsonify({"status": "ok", "predictions": predictions})
    else:
        return jsonify({"status": "no object detected!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

