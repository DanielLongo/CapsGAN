import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
import numpy as np
from utils import get_mnist_classifer
from utils import get_batch_of_images

def get_inception_score(generator, use_cuda=None):

	imgs = get_batch_of_images(generator, 8)
	net = get_mnist_classifer()
	net.eval()

	if torch.cuda.is_available() and use_cuda != False:
		net = net.cuda()
	elif (torch.cuda.is_available() == False) and use_cuda == False:
		# print("not using cuda")
		use_cuda = False
	else:
		# print("Cuda not availiabe but use_cuda is True")
		return

	batch_size = np.shape(imgs[0])[0] 
	assert(len(np.shape(imgs[0])) == 4), "Batches of imgs had incorrect number of dimensions. Expected 5. Recieved shape: " + str(np.shape(imgs))
	scores = []

	for batch in imgs:
		s = net(batch)
		scores +=  [s]
	# print("scores calculated")
	p_yx = F.softmax(torch.cat(scores, 0), 1)
	p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
	KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
	final_score = KL_d.mean()
	final_score = float(final_score.detach().cpu().numpy())
	# print("inception score", final_score)
	return final_score
