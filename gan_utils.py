import torch
from torch import nn
from utils import save_images
from utils import generate_noise
from metrics import get_inception_score

def create_optimizer(model, lr=.01, betas=None):
	if betas == None:
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
	return optimizer

def train_gan(generator, discriminator, image_loader, num_epochs, batch_size, cuda=True, g_lr=1e-3, d_lr=1e-3, filename_prefix="results", save_gen_images=False):
	if cuda:
		dtype = torch.cuda.FloatTensor
		generator.cuda()
		discriminator.cuda()
	else:
		dtype = torch.FloatTensor

	iters = 0
	d_optimizer = create_optimizer(discriminator, lr=d_lr, betas=(.5, .999))
	g_optimizer = create_optimizer(generator, lr=g_lr, betas=(.5, .999))
	BCELoss = nn.BCELoss()
	for epoch in range(num_epochs):
		for x, _ in image_loader:
			if x.shape[0] != batch_size:
				continue
				
			real_data = x.type(dtype)

			z = generate_noise(batch_size).type(dtype)
			fake_images = generator(z)
			g_result = discriminator(fake_images).squeeze()
			# g_cost = BCELoss(g_result, torch.ones(batch_size).type(dtype))
			g_cost = torch.mean(g_result)
			g_cost.backward()
			g_optimizer.step()
			g_optimizer.zero_grad()

			d_optimizer.zero_grad()
			z = generate_noise(batch_size).type(dtype)
			fake_images = generator(z)
			d_spred_fake = discriminator(fake_images).squeeze()
			d_cost_fake = BCELoss(d_spred_fake, torch.zeros(batch_size).type(dtype))
			d_spred_real = discriminator(real_data).squeeze()
			d_cost_real = BCELoss(d_spred_real, torch.ones(batch_size).type(dtype))
			# d_cost = d_cost_real + d_cost_fake
			d_cost = 0-  torch.mean(d_spred_real - d_spred_fake)
			d_cost.backward()
			d_optimizer.step()
			iters += 1
		if save_images:
			save_images(generator, epoch, iters, filename_prefix)
		print("Epoch", epoch, "Iter", iters)
		print("d_cost", d_cost)
		print("g_cost", g_cost)
		print("Inception Score", get_inception_score(generator))

	return discriminator, generator