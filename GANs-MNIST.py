from gan_utils import train_gan
from utils import get_mnist_data
from ConvDiscriminator import ConvDiscriminatorMNIST
from ConvGenerator import ConvGeneratorMNIST

G = ConvGeneratorMNIST()
D = ConvDiscriminatorMNIST()

train_data, test_data = get_mnist_data()

train_gan(G,D, train_data, 10, 64, save_gen_images=True)