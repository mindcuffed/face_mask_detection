from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.feature import hog
import joblib

app = Flask(__name__)

# load trained model
model = joblib.load("mask_model.pkl")

IMG_SIZE = (64, 64)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    return features

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    features = extract_features(img)
    prob = model.predict_proba([features])[0]

    mask_prob = prob[1]
    no_mask_prob = prob[0]

    if mask_prob > 0.5:
        result = "Mask"
        confidence = mask_prob
    else:
        result = "No Mask"
        confidence = no_mask_prob

    return jsonify({
        "prediction": result,
        "confidence": float(confidence)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)