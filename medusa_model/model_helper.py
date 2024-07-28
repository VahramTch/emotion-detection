import os
import cv2

class ModelHelper:

    def get_array_from_image_dataset(self, directory, classes, image_size):
        training_data = []
        for category in classes:
            path = os.path.join(directory, category)
            class_num = classes.index(category)
            print("Reading files in:", path)
            count = 0
            for image in os.listdir(path):
                if image.endswith('.jpg'):
                    image_array = cv2.imread(os.path.join(path, image))
                    resized_array = cv2.resize(image_array, (image_size, image_size))
                    training_data.append([resized_array, category])
                    count += 1
                
                if count == 200: break
        return training_data