from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as tt

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
model_state = torch.load("./models/emotion_detection_model_state.pth", map_location=torch.device('cpu'))
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(6), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)

model = ResNet(1, len(class_labels))
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