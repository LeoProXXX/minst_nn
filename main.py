from data_provider import DataProvider
from neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm


class Testing:
    @staticmethod
    def data_analysis():
        dp = DataProvider.load_from_folder('samples')

        train_y = dp.get_train_y()
        print("train data")
        y_value = np.zeros((1, 10))
        for i in range(10):
            print("occurance of ", i, "=", np.count_nonzero(train_y == i))
            y_value[0, i - 1] = np.count_nonzero(train_y == i)

        y_value = y_value.ravel()
        x_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.xlabel('klasa')
        plt.ylabel('liczba wystąpień klasy')
        plt.bar(x_value, y_value, 0.7, color='g')
        plt.show()

        test_y = dp.get_test_y()
        print("test data")
        y_value = np.zeros((1, 10))
        for i in range(10):
            print("occurance of ", i, "=", np.count_nonzero(test_y == i))
            y_value[0, i - 1] = np.count_nonzero(test_y == i)

        y_value = y_value.ravel()
        x_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.xlabel('klasa')
        plt.ylabel('liczba wystąpień klasy')
        plt.bar(x_value, y_value, 0.7, color='g')
        plt.show()

    @staticmethod
    def test_metrics():
        random.seed(10)

        dp = DataProvider.load_from_folder('samples')

        nn = NeuralNet(sizes=[784, 128, 10], epochs=10)
        nn.train(dp.get_train_x(), dp.get_hot_encoded_train_y(), dp.get_test_x(), dp.get_hot_encoded_test_y())

        scores = nn.compute_metrics(dp.get_test_x(), dp.get_hot_encoded_test_y())
        print('precyzja_makro: ', scores['precyzja_makro'])
        print('czulosc_makro: ', scores['czulosc_makro'])
        print('dokladnosc: ', scores['dokladnosc'])
        print('raport: ', scores['raport'])
        print('macierz_bledow: ', scores['macierz_bledow'])

    @staticmethod
    def show_images():
        random.seed(10)

        dp = DataProvider.load_from_folder('samples')

        nn = NeuralNet(sizes=[784, 128, 10], epochs=10)
        nn.train(dp.get_train_x(), dp.get_hot_encoded_train_y(), dp.get_test_x(), dp.get_hot_encoded_test_y())

        properly_classified, misclassified = nn.get_properly_classified_and_misclassified_images(dp.get_test_x(), dp.get_hot_encoded_test_y())

        print('properly classified')
        plt.imshow(properly_classified[0].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(properly_classified[1].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(properly_classified[2].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(properly_classified[3].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(properly_classified[4].reshape(28, 28), cmap=cm.binary)
        plt.show()

        print('missclasified')
        plt.imshow(misclassified[0].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(misclassified[1].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(misclassified[2].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(misclassified[3].reshape(28, 28), cmap=cm.binary)
        plt.show()
        plt.imshow(misclassified[4].reshape(28, 28), cmap=cm.binary)
        plt.show()


    @staticmethod
    def example():
        dp = DataProvider.load_from_folder('samples')

        nn = NeuralNet(sizes=[784, 128, 10])
        nn.train(dp.get_train_x(), dp.get_hot_encoded_train_y(), dp.get_test_x(), dp.get_hot_encoded_test_y())


# Testing.data_analysis()
# Testing.test_metrics()
Testing.show_images()
# Testing.example()
