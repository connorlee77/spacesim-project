import numpy as np
import os
import torch
import torch.optim as optim
import natsort 
import tqdm 
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = '../csv'

data = []
files = os.listdir(DATA_PATH)
files = natsort.natsorted(files)
for file in files:
	filepath = os.path.join(DATA_PATH, file)
	f = np.loadtxt(filepath, delimiter=',')
	data.append(f)
data = np.vstack(data)

xdata = np.zeros((len(data), 3))
xdata = data[:,3:6]

ydata = data[:,9:12]
x_train, x_test, y_train, y_test = xdata[:7000,:], xdata[7000:,:], ydata[:7000,:], ydata[7000:,:]

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
batch_size, D_in, H, D_out = 64, len(xdata[0]), 5, 3

def getModel():
	model = torch.nn.Sequential(
		torch.nn.Linear(D_in, H),
		torch.nn.ReLU(),
		torch.nn.Linear(H, H),
		torch.nn.ReLU(),
		torch.nn.Linear(H, D_out),
	)

	model.load_state_dict(torch.load('../weightsrand.pt'))
	model.eval()

	return model

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.

if __name__ == '__main__':
	predict = getModel()
	with torch.no_grad():
		test_loss = 0
		for x, y in test_loader:
			y_pred = predict(x)
			print(y_pred.numpy()[0])
