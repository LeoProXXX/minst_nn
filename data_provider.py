from mnist import MNIST
import numpy as np


class DataProvider():
    def __init__(self, train_images_np, train_labels_np, test_images_np, test_labels_np):
        self.train_images_np = train_images_np
        self.train_labels_np = train_labels_np
        self.train_labels_hot_encoded = DataProvider.one_hot_encode(train_labels_np)

        self.test_images_np = test_images_np
        self.test_labels_np = test_labels_np
        self.test_labels_hot_encoded = DataProvider.one_hot_encode(test_labels_np)

    @staticmethod
    def load_from_folder(folder_name):
        mndata = MNIST(folder_name)
        mndata.gz = True

        train_images, train_labels = mndata.load_training()
        train_images_np = (np.array(train_images) / 255).astype('float32')
        train_labels_np = np.array(train_labels)

        test_images, test_labels = mndata.load_testing()
        test_images_np = (np.array(test_images) / 255).astype('float32')
        test_labels_np = np.array(test_labels)

        return DataProvider(train_images_np, train_labels_np, test_images_np, test_labels_np)

    def get_train_x(self):
        return self.train_images_np

    def get_train_y(self):
        return self.train_labels_np

    def get_hot_encoded_train_y(self):
        return self.train_labels_hot_encoded

    def get_test_x(self):
        return self.test_images_np

    def get_test_y(self):
        return self.test_labels_np

    def get_hot_encoded_test_y(self):
        return self.test_labels_hot_encoded

    @staticmethod
    def one_hot_encode(labels):
        num_of_labels = len(labels)
        num_of_classes = np.max(labels) + 1

        one_hot_labels = np.zeros((num_of_labels, num_of_classes))

        for i in range(num_of_labels):
            one_hot_labels[i, labels[i]] = 1

        return one_hot_labels
