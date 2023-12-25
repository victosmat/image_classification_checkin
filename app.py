from flask import Flask, jsonify, request
import base64
import os
from flask_cors import CORS, cross_origin
from src.face_rec_image import recognize_face
from flask import jsonify
import random

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/recognize_face', methods=['POST'])
@cross_origin()
def recognizeFace():
    data = request.get_json()
    print(data)
    if 'image' in data:
        image_data = data['image']
        image_data = image_data.split(',')[1] 
        image = base64.b64decode(image_data)

        file_path = os.path.join(UPLOAD_FOLDER, 'captured_image.jpg')
        with open(file_path, 'wb') as file:
            file.write(image)

        employeeId, probability = recognize_face(file_path)

        if probability > 70:
            image_employeeId = str(random.randint(1, 1000)) + '.jpg'
            file_path = os.path.join('Dataset/FaceData/raw', employeeId, image_employeeId)
            with open(file_path, 'wb') as file:
                image = file.write(image)
            return jsonify({'employeeId': employeeId, 'probability': probability, 'isSave': True})

        if employeeId is not None and probability is not None:
            return jsonify({'employeeId': employeeId, 'probability': probability, 'isSave': False})
        else:
            return 'Recognition failed'
    else:
        return 'No image found in request'
    

@app.route('/save_image', methods=['POST'])
@cross_origin()
def saveImage():
    data = request.get_json()
    print(data)
    employeeId = data['employeeId']
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = base64.b64decode(image_data)
    file_path = os.path.join('Dataset/FaceData/raw', employeeId, 'image.jpg')
    try:
        with open(file_path, 'wb') as file:
            file.write(image)
        return jsonify({'data': True})
    except Exception as e:
        print(e)
        return jsonify({'data': False})

if __name__ == "__main__":
    app.run(debug=True)