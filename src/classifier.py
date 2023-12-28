import tensorflow as tf
import numpy as np
from src.facenet import *
import os
import math
import pickle
from sklearn.svm import SVC

def train_model_from_dataset():
    data_dir = 'Dataset/FaceData/processed' # Đường dẫn đến thư mục chứa các thư mục ảnh của các người
    model = 'Models/20180402-114759.pb' # Đường dẫn đến file model
    classifier_filename = 'Models/facemodel.pkl' # Đường dẫn đến file classifier
    batch_size = 1000 # Batch size
  
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess: # Khởi tạo một graph mới và một session mới
        np.random.seed(seed=666)  # Khởi tạo seed cho numpy
        dataset = get_dataset(data_dir) # Lấy ra danh sách các thư mục ảnh
        print("#"*100)
        print(dataset)
        paths, labels = get_image_paths_and_labels(dataset) # Lấy ra danh sách các ảnh và nhãn tương ứng của các ảnh
        
        print('Number of classes: %d' % len(dataset))  # In ra số lượng các thư mục ảnh của các người
        print('Number of images: %d' % len(paths)) # In ra số lượng các ảnh
        
        load_model(model) # Load model
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0") # Lấy tensor input
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0") # Lấy tensor embeddings
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0") # Lấy tensor phase_train
        embedding_size = embeddings.get_shape()[1] # Lấy ra kích thước của vector embedding
        
        print('Calculating features for images') # In ra thông báo
        nrof_images = len(paths) # Lấy ra số lượng các ảnh
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size)) # Tính số lượng batch mỗi epoch
        emb_array = np.zeros((nrof_images, embedding_size)) # Khởi tạo mảng chứa các vector embedding
        
        for i in range(nrof_batches_per_epoch): # Duyệt qua từng batch
            start_index = i * batch_size # Tính chỉ số bắt đầu của batch
            end_index = min((i + 1) * batch_size, nrof_images) # Tính chỉ số kết thúc của batch
            paths_batch = paths[start_index:end_index] # Lấy ra danh sách các ảnh của batch
            images = load_data(paths_batch, False, False, 160) # Load ảnh và resize ảnh về kích thước 160x160
            feed_dict = {images_placeholder: images, phase_train_placeholder: False} # Tạo feed_dict
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict) # Tạo vector embedding cho các ảnh của batch
        
        classifier_filename_exp = os.path.expanduser(classifier_filename) # Tạo đường dẫn tuyệt đối đến file classifier
        print('Training classifier') # In ra thông báo
        model = SVC(kernel='linear', probability=True) # Khởi tạo model
        model.fit(emb_array, labels) # Train model
    
        class_names = [cls.name.replace('_', ' ') for cls in dataset] # Lấy ra tên của các thư mục ảnh của các người
        with open(classifier_filename_exp, 'wb') as outfile: # Mở file classifier
            pickle.dump((model, class_names), outfile) # Lưu model và class_names vào file classifier
        print('Saved classifier model to file "%s"' % classifier_filename_exp) # In ra thông báo

if __name__ == '__main__':
    train_model_from_dataset()