import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.datasets
from torchvision import transforms
plt.rcParams['image.cmap'] = 'gray'

def save_images(generator, epoch, i, filename_prefix):
	fig = plt.figure(figsize=(10, 10))
	gs = gridspec.GridSpec(10, 10)
	gs.update(wspace=.05, hspace=.05)
	z = generate_noise(100)
	if next(generator.parameters()).is_cuda:
		z = z.type(torch.cuda.FloatTensor)
	images_fake = generator(z)
	images_fake = images_fake.data.data.cpu().numpy()
	for img_num, sample in enumerate(images_fake):
		ax = plt.subplot(gs[img_num])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap='Greys_r')

	filename = filename_prefix + str(epoch) + "-" + str(i) 
	plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
	plt.close(fig)

def generate_noise(batch_size, dim=100):
	noise = torch.randn(batch_size, dim)
	return noise

def get_mnist_data(batch_size=64):
	transform = transforms.ToTensor()
	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
	mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True)
	return train_loader, test_loader