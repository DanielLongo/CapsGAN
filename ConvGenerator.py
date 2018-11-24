import torch 
from torch import nn

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(100, 128, [2,2], stride=[1,1]),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(128, 256, [3,3], stride=[1,1]),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.deconv3 = nn.Sequential(
			nn.ConvTranspose2d(256, 256, [4,4], stride=[2,2], padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.deconv4 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, [4,4], stride=[2,2], padding=1),
			nn.BatchNorm2d(128))
		self.deconv5 = nn.Sequential(
			nn.ConvTranspose2d(128, 1, [4,4], stride=[2,2], padding=1),
			nn.Tanh())

	def forward(self, x):
		out = self.deconv1(x)
		out = self.deconv2(out)
		out = self.deconv3(out)
		out = self.deconv4(out)
		out = self.deconv5(out)
		return out