import torch 
from torch import nn

class ConvGeneratorMNIST(nn.Module):
	def __init__(self, d=100):
		super(ConvGeneratorMNIST, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(d, 1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024)
		)
		self.fc2 = nn.Sequential(
			nn.Linear(1024, (7*7*128)),
			nn.ReLU(),
			nn.BatchNorm1d(7*7*128)
		)
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, [4,4], stride=[2,2], padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64)
		)
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(64, 1, [4,4], stride=[2,2], padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = out.view(out.shape[0], 128, 7, 7)
		out = self.deconv1(out)
		out = self.deconv2(out)
		return out