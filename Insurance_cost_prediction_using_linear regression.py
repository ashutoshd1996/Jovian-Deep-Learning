import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

DATASET_URL = "https://gist.github.com/BirajCoder/5f068dfe759c1ea6bdfce9535acdb72d/raw/c84d84e3c80f93be67f6c069cbdc0195ec36acbd/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')

dataframe_raw = pd.read_csv(DATA_FILENAME)
# print(dataframe_raw.head())

your_name = "Ashutosh"

def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.
    # scale target
    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['region'], axis=1)
    return dataframe

dataframe = customize_dataset(dataframe_raw, your_name)
print(dataframe.head())

num_rows = dataframe.shape[0]
print('\n'+str(num_rows))

num_cols = dataframe.shape[1]
print(num_cols)

input_cols = list(dataframe.drop(['charges'], axis=1).columns)
print(input_cols)

categorical_cols = list(set(input_cols) - set(dataframe._get_numeric_data().columns))
print(categorical_cols)

output_cols = ['charges']

print(dataframe[output_cols].describe())


def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays

    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array


inputs_array, targets_array = dataframe_to_arrays(dataframe)
print(inputs_array, targets_array)

from sklearn.preprocessing import StandardScaler
targets_scaler = StandardScaler()
targets_array = targets_scaler.fit_transform(targets_array)

inputs = torch.tensor(torch.from_numpy(inputs_array), dtype=torch.float32)
targets = torch.tensor(torch.from_numpy(targets_array), dtype=torch.float32)

dataset = TensorDataset(inputs, targets)

val_percent = 0.18
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 64

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = len(input_cols)
output_size = len(output_cols)

print(input_size, output_size, batch_size)


class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # fill this (hint: use input_size & output_size defined above)

    def forward(self, xb):
        out = self.linear(xb)  # fill this
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calcuate loss
        loss = F.mse_loss(out, targets)  # fill this
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.mse_loss(out, targets)  # fill this
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


model = InsuranceModel()
print(list(model.parameters()))
result = evaluate(model, val_loader) # Use the the evaluate function
print(result)
epochs = 1000
lr = 3e-4
history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 1e-4
history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000
lr = 9e-5
history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 500
lr = 5e-5
history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 500
lr = 1e-5
history5 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = history5


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)                # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

input, target = val_ds[0]
predict_single(input, target, model)
input, target = val_ds[10]
predict_single(input, target, model)
input, target = val_ds[23]
predict_single(input, target, model)
