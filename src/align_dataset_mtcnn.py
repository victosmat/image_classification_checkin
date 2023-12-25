from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import random
from time import sleep
import cv2
from face_extractor import extract_face  # Import hàm extract_face


def main(args=None):
    input_dir = 'Dataset/FaceData/raw'
    output_dir = 'Dataset/FaceData/processed'
    random_order = True

    detect_multiple_faces = False  # Sử dụng giá trị mặc định cho detect_multiple_faces

    sleep(random.random())

    if not os.path.exists(output_dir):  # Kiểm tra xem thư mục output có tồn tại hay không
        os.makedirs(output_dir)  # Nếu không tồn tại thì tạo thư mục output

    dataset = []
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            dataset.append(os.path.join(root, dir))

    print('Loading face detection model')

    random_key = np.random.randint(0, high=99999)  # Tạo một số ngẫu nhiên trong khoảng từ 0 đến 99999
    bounding_boxes_filename = os.path.join(output_dir,
                                           'bounding_boxes_%05d.txt' % random_key)  # Tạo tên file bounding_boxes

    with open(bounding_boxes_filename, "w") as text_file:  # Mở file bounding_boxes
        nrof_images_total = 0  # Khởi tạo biến nrof_images_total
        nrof_successfully_aligned = 0  # Khởi tạo biến nrof_successfully_aligned
        if random_order:  # Nếu random_order = True
            random.shuffle(dataset)  # Xáo trộn danh sách các thư mục trong dataset
        for cls in dataset:  # Duyệt qua từng thư mục trong dataset
            output_class_dir = os.path.join(output_dir, os.path.basename(cls))  # Tạo đường dẫn đến thư mục output
            if not os.path.exists(output_class_dir):  # Kiểm tra xem thư mục output có tồn tại hay không
                os.makedirs(output_class_dir)  # Nếu không tồn tại thì tạo thư mục output
                if random_order:  # Nếu random_order = True
                    random.shuffle(os.listdir(cls))  # Xáo trộn danh sách các ảnh trong thư mục
            for image_path in os.listdir(cls):  # Duyệt qua từng ảnh trong thư mục
                image_path = os.path.join(cls, image_path)  # Đường dẫn đến ảnh
                nrof_images_total += 1  # Tăng biến nrof_images_total lên 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]  # Lấy tên file ảnh
                output_filename = os.path.join(output_class_dir,
                                               filename + '.png')  # Tạo đường dẫn đến file ảnh trong thư mục output
                print(image_path)  # In ra đường dẫn đến ảnh
                if not os.path.exists(
                        output_filename):  # Kiểm tra xem file ảnh trong thư mục output có tồn tại hay không
                    try:  # Thử đọc ảnh
                        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Đọc ảnh
                        if img is None:  # Kiểm tra xem ảnh có đọc được hay không
                            raise ValueError("The image cannot be read.")  # Nếu không đọc được thì ném ra lỗi
                    except (IOError, ValueError, IndexError) as e:  # Nếu có lỗi xảy ra
                        errorMessage = '{}: {}'.format(image_path, e)  # Lưu thông tin lỗi
                        print(errorMessage)  # In ra thông tin lỗi
                        text_file.write('%s\n' % (output_filename))  # Ghi thông tin lỗi vào file bounding_boxes
                        continue  # Bỏ qua ảnh hiện tại

                    # Sử dụng hàm extract_face để cắt mặt từ ảnh
                    faces = extract_face(image_path)

                    for i, face in enumerate(faces):  # Duyệt qua từng khuôn mặt được cắt
                        nrof_successfully_aligned += 1  # Tăng biến nrof_successfully_aligned lên 1
                        filename_base, file_extension = os.path.splitext(output_filename)  # Lấy tên file và đuôi file
                        if detect_multiple_faces:  # Nếu detect_multiple_faces = True
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)  # Tạo tên file mới
                        else:  # Nếu detect_multiple_faces = False
                            output_filename_n = "{}{}".format(filename_base, file_extension)  # Giữ nguyên tên file
                        if face is not None and face.size > 0:
                            cv2.imwrite(output_filename_n, face)  # Lưu ảnh vào thư mục output
                        else:
                            print("Error: Invalid or empty face image.")
                        text_file.write('%s %d %d %d %d\n' % (
                            output_filename_n, 0, 0, 0, 0))  # Ghi thông tin bounding box vào file bounding_boxes
                else:
                    print('Output file "%s" already exists, skipping.' % output_filename)

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


if __name__ == '__main__':
    main()
