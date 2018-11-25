import torch 
from torch import nn

class ConvDiscriminatorMNIST(nn.Module):
	def __init__(self):
		super(ConvDiscriminatorMNIST, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 32, [5,5], stride=[1,1]),
			nn.LeakyReLU(negative_slope=.01),
			nn.MaxPool2d([2,2], stride=[2,2])
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, [5,5], stride=[1,1]),
			nn.LeakyReLU(negative_slope=.01),
			nn.MaxPool2d([2,2], stride=[2,2])
		)
		self.fc1 = nn.Sequential(
			nn.Linear((64*4*4), (64*4*4)),
			nn.LeakyReLU(negative_slope=.01)
		)
		self.fc2 = nn.Sequential(
			nn.Linear((64*4*4), 1),
			nn.Sigmoid()
		)


	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out
