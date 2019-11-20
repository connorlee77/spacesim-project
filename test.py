import numpy as np
import os
import torch
import torch.optim as optim
import natsort 
import tqdm 
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = 'csv'

data = []
files = os.listdir(DATA_PATH)
files = natsort.natsorted(files)
for file in files:
	filepath = os.path.join(DATA_PATH, file)
	print(file)
	f = np.loadtxt(filepath, delimiter=',')
	data.append(f)
data = np.vstack(data)

xdata = np.zeros((len(data), 3))
xdata = data[:,3:6]

ydata = data[:,9:12]
x_train, x_test, y_train, y_test = xdata[:7000,:], xdata[7000:,:], ydata[:7000,:], ydata[7000:,:]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
batch_size, D_in, H, D_out = 64, len(xdata[0]), 5, 3

# Create random Tensors to hold inputs and outputs
tensor_data = torch.from_numpy(x_train).float()
tensor_target = torch.from_numpy(y_train).float()

tensor_data_val = torch.from_numpy(x_test).float()
tensor_target_val = torch.from_numpy(y_test).float()

train_dataset = data_utils.TensorDataset(tensor_data, tensor_target)
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = data_utils.TensorDataset(tensor_data_val, tensor_target_val)
test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out),
)
model.load_state_dict(torch.load('weightsrand.pt'))
model.eval()

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

with torch.no_grad():
	test_loss = 0
	for x, y in test_loader:
		y_pred = model(x)

		print("Predicted: ", y_pred)
		print("Actual: ", x*0.1)
		print()

		test_loss += loss_fn(y_pred, x*0.1).item()
	print(test_loss/ len(test_loader))
