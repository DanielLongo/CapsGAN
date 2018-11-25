from gan_utils import train_gan
from utils import get_mnist_data
from ConvDiscriminator import ConvDiscriminatorMNIST
from ConvGenerator import ConvGeneratorMNIST
from CapsDiscriminator import CapsDiscriminatorMNIST

train_data, test_data = get_mnist_data()

#DCGAN -> G: Conv D: Conv
G = ConvGeneratorMNIST()
D = ConvDiscriminatorMNIST()

# train_gan(G,D, train_data, 10, 64, save_gen_images=True)

#CapsGAN -> G: Conv D: Caps
G = ConvGeneratorMNIST()
D = CapsDiscriminatorMNIST() 

train_gan(G,D, train_data, 10, 64, save_gen_images=True)