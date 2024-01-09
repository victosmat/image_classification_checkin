from datetime import datetime
from flask import Flask, jsonify, request
import base64
import os
from flask_cors import CORS, cross_origin
from src.face_rec_image import recognize_face
from flask import jsonify
from src.align_dataset_mtcnn import extract_face_to_process
from src.classifier import train_model_from_dataset
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import random

sentry_sdk.init(
    dsn = "http://2d93a0f392f343ffb5c0cecfc5393448@localhost:9000/4",
    integrations = [FlaskIntegration()]
)

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/recognize_face", methods=["POST"])
@cross_origin()
def recognizeFace():
    data = request.get_json()
    print(data)
    if "image" in data:
        image_data = data["image"]
        image_data = image_data.split(",")[1]
        image = base64.b64decode(image_data)

        file_path = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
        with open(file_path, "wb") as file:
            file.write(image)

        employeeId, probability = recognize_face(file_path)

        if probability > 70:
            current_date = datetime.now()
            image_employeeId = current_date.strftime("%H-%M-%S_%d-%m-%Y") + ".jpg"
            file_path = os.path.join("Dataset/FaceData/raw", employeeId, image_employeeId)

            with open(file_path, "wb") as file:
                image = file.write(image)
            return jsonify(
                {"employeeId": employeeId, "probability": probability, "isSave": True}
            )

        if employeeId is not None and probability is not None:
            return jsonify(
                {"employeeId": employeeId, "probability": probability, "isSave": False}
            )
        else:
            return "Recognition failed"
    else:
        return "No image found in request"


@app.route("/save_image", methods=["POST"])
@cross_origin()
def saveImage():
    data = request.get_json()
    print(data)
    employeeId = data["employeeId"]
    current_date = datetime.now()
    image_employeeId = current_date.strftime("%H-%M-%S_%d-%m-%Y") + ".jpg"
    file_path = os.path.join("Dataset/FaceData/raw", employeeId, image_employeeId)

    if not os.path.exists(file_path):
        return jsonify({"error": "Folder not found"})

    file_path_upload = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
    with open(file_path_upload, "rb") as file:
        image = file.read()
    try:
        with open(file_path, "wb") as file:
            file.write(image)
        return jsonify({"data": True})
    except Exception as e:
        print(e)
        return jsonify({"data": False})


@app.route("/train_model", methods=["GET"])
@cross_origin()
def clearUploadFolder():
    try:
        extract_face_to_process()
        train_model_from_dataset()
        return jsonify({"data": True})
    except Exception as e:
        print(e)
        return jsonify({"data": False})


@app.route("/get_images_base64", methods=["POST"])
@cross_origin()
def getImagesBase64():
    data = request.get_json()
    print(data)
    employeeId = data["employeeId"]
    month = data["month"]
    year = data["year"]
    images_base64 = []
    folder_path = os.path.join("Dataset/FaceData/raw", employeeId)

    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"})

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                file_name = os.path.splitext(filename)[0]
                month_file = file_name.split("_")[1].split("-")[1].lstrip("0")
                year_file = file_name.split("_")[1].split("-")[2].lstrip("0")
                if month_file != month or year_file != year:
                    continue

                image_data = file.read()
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                
                images_base64.append( {"nameFile": file_name, "imageBase64": encoded_image})

    return jsonify({"images_base64": images_base64})


@app.route("/delete_image", methods=["POST"])
@cross_origin()
def deleteImage():
    data = request.get_json()
    print(data)
    employeeId = data["employeeId"]
    file_name = data["nameFile"]

    file_path = os.path.join("Dataset/FaceData/raw", employeeId, file_name + ".jpg")
    if not os.path.exists(file_path):
        return jsonify({"error": "Folder not found"})

    try:
        os.remove(file_path)
        return jsonify({"data": True})
    except Exception as e:
        print(e)
        return jsonify({"data": False})


@app.route("/save_list_images", methods=["POST"])
@cross_origin()
def saveListImages():
    data = request.get_json()
    print(data)
    employeeId = data["employeeId"]
    images = data["images"]

    folder_path = os.path.join("Dataset/FaceData/raw", employeeId)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for image in images:
        current_date = datetime.now()
        strRandom = random.randint(100000, 999999).__str__()
        image_employeeId = current_date.strftime("%H-%M-%S_%d-%m-%Y_") + strRandom + ".jpg"
        file_path = os.path.join(folder_path, image_employeeId)
        image_data = base64.b64decode(image.split(",")[1])
        with open(file_path, "wb") as file:
            file.write(image_data)

    return jsonify({"data": True})

@app.route("/get_avatar", methods=["POST"])
@cross_origin()
def getAvatar():
    data = request.get_json()
    print(data)
    employeeId = data["employeeId"]
    file_path = os.path.join("Dataset/FaceData/raw", employeeId)
    if not os.path.exists(file_path):
        return jsonify({"error": "Folder not found"})

    for filename in os.listdir(file_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(file_path, filename)
            with open(file_path, "rb") as file:
                image_data = file.read()
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                return jsonify({"imageBase64": encoded_image})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
