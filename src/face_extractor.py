import cv2
import numpy as np
import os


def extract_face(image_path):
    DEFAULT_PROTO = os.path.expanduser("deploy.prototxt.txt") # Đường dẫn đến file deploy.prototxt.txt
    DEFAULT_MODEL = os.path.expanduser("res10_300x300_ssd_iter_140000.caffemodel") # Đường dẫn đến file res10_300x300_ssd_iter_140000.caffemodel
    DEFAULT_CONFIDENCE = 0.7 # Giá trị mặc định của confidence
    image_size = 160 # Kích thước ảnh đầu vào
    margin = 32 # Kích thước margin

    frame = cv2.imread(image_path) # Đọc ảnh đầu vào
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Chuyển đổi sang định dạng màu RGB

    net = cv2.dnn.readNetFromCaffe(DEFAULT_PROTO, DEFAULT_MODEL) # Load model

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) # Tạo blob
    net.setInput(blob) # Set blob vào model
    detections = net.forward() # Feed-forward

    faces = [] # Khởi tạo mảng chứa các khuôn mặt

    for i in range(0, detections.shape[2]): # Duyệt qua từng detection
        confidence = detections[0, 0, i, 2] # Lấy ra confidence của detection
 
        if confidence > DEFAULT_CONFIDENCE: # Nếu confidence > DEFAULT_CONFIDENCE
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]) # Lấy ra tọa độ của bounding box
            (startX, startY, endX, endY) = box.astype("int") # Chuyển tọa độ bounding box về kiểu int

            margin_x = int((endX - startX) * margin / 100)  # Tính margin theo chiều x
            margin_y = int((endY - startY) * margin / 100) # Tính margin theo chiều y
            startX = max(0, startX - margin_x) # Tính tọa độ bắt đầu của bounding box
            startY = max(0, startY - margin_y) # Tính tọa độ bắt đầu của bounding box
            endX = min(frame.shape[1], endX + margin_x) # Tính tọa độ kết thúc của bounding box
            endY = min(frame.shape[0], endY + margin_y) # Tính tọa độ kết thúc của bounding box
 
            face = frame[startY:endY, startX:endX] # Cắt khuôn mặt từ ảnh đầu vào
            face = cv2.resize(face, (image_size, image_size))  # Resize khuôn mặt về kích thước 160x160
            faces.append(face)  # Thêm khuôn mặt vào mảng chứa các khuôn mặt

    return faces
