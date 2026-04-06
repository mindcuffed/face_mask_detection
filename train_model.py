import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import xml.etree.ElementTree as ET

# ----------------------------
# SETTINGS (MUST MATCH DETECTION FILE)
# ----------------------------
IMG_SIZE = (128, 128)

images_path = "dataset/images"
annotations_path = "dataset/annotations"

feature_list = []
label_list = []

print("Reading dataset...")

for xml_file in os.listdir(annotations_path):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(annotations_path, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(images_path, filename)

    img = cv2.imread(img_path)
    if img is None:
        continue

    for obj in root.findall('object'):
        label = obj.find('name').text

        if label == 'with_mask':
            y = 1
        elif label == 'without_mask':
            y = 0
        else:
            continue

        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        face = img[ymin:ymax, xmin:xmax]
        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, IMG_SIZE)

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        feature_list.append(features)
        label_list.append(y)

print("Total samples:", len(feature_list))

X = np.array(feature_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    C=50
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc * 100)

joblib.dump(model, "mask_model.pkl")
print("Model saved.")