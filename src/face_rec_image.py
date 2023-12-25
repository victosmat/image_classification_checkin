from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
import numpy as np
import cv2

def recognize_face(image_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', help='Path of the cropped face image you want to test on.') # Sửa mô tả của tham số
    # args = parser.parse_args()

    IMAGE_PATH = image_path
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file) # load model và class_names từ file
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default(): # Khởi tạo một graph mới

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6) # Cấu hình GPU
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) # Khởi tạo một session mới

        with sess.as_default():

            print('Loading feature extraction model')
            load_facenet_model(FACENET_MODEL_PATH)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") # Lấy tensor input
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # Lấy tensor embeddings
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0") # Lấy tensor phase_train

            face_image = cv2.imread(IMAGE_PATH)  # Đọc ảnh đầu vào
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang định dạng màu RGB

            scaled = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_CUBIC) # Resize ảnh về kích thước 160x160
            scaled = prewhiten(scaled)  # Chuẩn hóa ảnh
            scaled_reshape = scaled.reshape(-1, 160, 160, 3) # Reshape ảnh về dạng 4 chiều

            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False} # Tạo feed_dict
            emb_array = sess.run(embeddings, feed_dict=feed_dict) # Tạo vector embedding cho ảnh đầu vào

            predictions = model.predict_proba(emb_array) # Dự đoán ảnh đầu vào thuộc lớp nào
            best_class_indices = np.argmax(predictions, axis=1) # Lấy ra lớp có xác suất cao nhất
            best_probability = predictions[np.arange(len(best_class_indices)), best_class_indices][0] # Lấy ra xác suất của lớp có xác suất cao nhất
            employeeId = class_names[best_class_indices[0]] # Lấy ra tên của lớp có xác suất cao nhất
            print("Name: {}, Probability: {}".format(employeeId, best_probability))

            return employeeId, best_probability * 100

def load_facenet_model(model_path):
    with tf.io.gfile.GFile(model_path, 'rb') as file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')
    print('FaceNet model loaded successfully')

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  