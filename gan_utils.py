import torch
from torch import nn
from utils import save_images

def train_gan(discriminator, generator, image_loader, num_epochs, batch_size, cuda=True, g_lr=1e-3, d_lr=1e-3, filename_prefix="results", save_images=False):
    if cuda:
        dtype = torch.cuda.FloatTensor
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

            z = generate_nosie(batch_size)
            fake_images = generator(z)
            g_result = discriminator(fake_images).squeeze()
            g_cost = BCELoss(g_result, torch.ones(batch_size))
            g_cost.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()

            d_optimizer.zero_grad()
            z = generate_nosie(batch_size)
            fake_images = generator(z)
            d_spred_fake = discriminator(fake_images).squeeze()
            d_cost_fake = BCELoss(d_spred_fake, torch.zeros(batch_size))
            d_spred_real = discriminator(real_data).squeeze()
            d_cost_real = BCELoss(d_spred_real, torch.ones(batch_size))
            d_cost = d_cost_real + d_cost_fake
            d_cost.backward()
            d_optimizer.step()
            iters += 1
        if save_images:
            save_images(generator, epoch, iters, filename_prefix)
        print("Epoch", epoch, "Iter", iters)
        print("d_cost", d_cost)
        print("g_cost", g_cost)

    return discriminator, generator