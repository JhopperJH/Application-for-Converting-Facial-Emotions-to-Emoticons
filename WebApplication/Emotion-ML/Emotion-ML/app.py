from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
model_state = torch.load("./models/model2.pth", map_location=torch.device('cpu'))
class_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class EmotionNet(nn.Module):
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        super(EmotionNet, self).__init__()
        self.num_of_channels = num_of_channels
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(nn.Linear(6*6*128, 64), nn.ELU(True), nn.Dropout(p=0.5), nn.Linear(64, num_of_classes))

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=True)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels, cfg):
        layers = []
        for X in cfg:
            if X == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, X, kernel_size=3, padding=1), nn.BatchNorm2d(X), nn.ELU(inplace=True)]
                in_channels = X
        return nn.Sequential(*layers)

model = EmotionNet(1, len(class_labels))
model.load_state_dict(model_state)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']
    image_np = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    predictions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = tt.functional.to_pil_image(roi_gray)
            roi = tt.functional.to_grayscale(roi)
            roi = tt.ToTensor()(roi).unsqueeze(0)

            # make a prediction on the ROI
            tensor = model(roi)
            pred = torch.max(tensor, dim=1)[1].tolist()
            label = class_labels[pred[0]]
            predictions.append({'label': label})
        else:
            predictions.append({'label': 'No Face Found'})

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)