from gan_utils import train_gan
from utils import get_mnist_data
from ConvDiscriminator import ConvDiscriminatorMNIST
from ConvGenerator import ConvGeneratorMNIST
from CapsDiscriminator import CapsDiscriminatorMNIST
from metrics import get_inception_score

batch_size = 128
epochs = 10
save_images = True
train_data, test_data = get_mnist_data(batch_size=batch_size)

#DCGAN -> G: Conv D: Conv
G = ConvGeneratorMNIST()
D = ConvDiscriminatorMNIST()

train_gan(G,D, train_data, epochs, batch_size, save_gen_images=save_images, filename_prefix="ConvConv/train")

#CapsGAN -> G: Conv D: Caps
G = ConvGeneratorMNIST()
D = CapsDiscriminatorMNIST(input_size=[1, 28, 28], classes=1, routings=3) 

train_gan(G,D, train_data, epochs, batch_size, save_gen_images=save_images, filename_prefix="ConvCaps/train")
print("score:", get_inception_score(G))