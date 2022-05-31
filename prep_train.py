import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
pas_ids = test_data["PassengerId"].tolist()


def transform_data(df):
    one_hot_columns = ["HomePlanet", "Destination", "CryoSleep", "VIP", "CabinDeck", "CabinSide"]
    df = df.drop(['PassengerId','Name'], axis=1)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric_tmp = df.select_dtypes(include=numerics)
    categ_tmp = df.select_dtypes(exclude=numerics)

    for col in numeric_tmp.columns:
        df[col] = df[col].fillna(value=df[col].mean())

    for col in categ_tmp.columns:
        df[col] = df[col].fillna(value=df[col].mode()[0])

    df['CabinDeck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['CabinNum'] = df['Cabin'].str.split('/', expand=True)[1]
    df['CabinSide'] = df['Cabin'].str.split('/', expand=True)[2]

    df.drop(columns=['Cabin', "CabinNum"], inplace=True)

    for column in one_hot_columns:
        tempdf = pd.get_dummies(df[column], prefix=column)
        df = pd.merge(
            left=df,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        df = df.drop(columns=column)
    features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df



def transform_train_data(df):
    tempdf = pd.get_dummies(df["Transported"], prefix="Transported")
    df = pd.merge(
        left=df,
        right=tempdf,
        left_index=True,
        right_index=True,
    )
    df = df.drop(columns="Transported")
    return df

train_data = transform_data(train_data)
train_data = transform_train_data(train_data).to_numpy(dtype=np.float32)

X_train = train_data[:, :-2]
Y_train = train_data[:, -2:]
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)


Y_train = np.reshape(Y_train, (len(Y_train), 2))

Y_val = np.reshape(Y_val, (len(Y_val), 2))
X_test = transform_data(test_data).to_numpy(dtype=np.float32)

tensor_X_train = torch.Tensor(X_train)
tensor_Y_train = torch.Tensor(Y_train)
tensor_X_val = torch.Tensor(X_val)
tensor_Y_val = torch.Tensor(Y_val)
tensor_X_test = torch.Tensor(X_test)

train_dataset = TensorDataset(tensor_X_train, tensor_Y_train)
val_dataset = TensorDataset(tensor_X_val, tensor_Y_val)

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 32)
        self.l4 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.2)
        #self.b1 = nn.BatchNorm1d(64)
        #self.b2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.silu(self.dropout(self.l1(x)))
        x = F.silu(self.dropout(self.l2(x)))
        x = F.silu(self.dropout(self.l3(x)))
        x = F.sigmoid(self.l4(x))
        return x

input_size = len(X_train[1, :])

model = NeuralNet(input_size).float()
optimizer = optim.Adam(model.parameters(), lr=0.0004)
loss = nn.BCELoss()

epochs = 150
min_valid_loss = np.inf
train_loss = 0


for i in range(epochs):
    for features, labels in train_dataloader:
        optimizer.zero_grad()


        y_pred = model.forward(features)
        single_loss = loss(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        train_loss += single_loss.item()
        #train_l.append(single_loss.item())

    valid_loss = 0
    with torch.no_grad():
        for vfeatures, vlabels in val_dataloader:
            vy_pred = model.forward(vfeatures)

            valid_loss = loss(vy_pred, vlabels)

            #val_l.append(valid_loss.item())

        print(f'Epoch: {i+1} Training loss: {train_loss / len(train_dataloader)} Validation loss: {valid_loss / len(val_dataloader)}')
        if min_valid_loss > valid_loss.item():
            min_valid_loss = valid_loss.item()
            print(f'Best valid loss: {valid_loss.item()}, saving model')
            torch.save(model.state_dict(), 'saved_model.pth')

    train_loss = 0
    model.train()
    torch.save(model.state_dict(), 'last_model.pth')



model = NeuralNet(input_size)
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()
preds = []

count = X_test.shape[0]
result_np = []

for idx in range(0, count):
    pred = model(torch.Tensor(X_test[idx]))
    predicted_class = np.argmax(pred.detach().numpy())
    result_np.append(predicted_class)

data_tuples = list(zip(pas_ids, result_np))
submission = pd.DataFrame(data_tuples, columns=["PassengerId", "Transported"])
submission = submission.replace(0, False)
submission = submission.replace(1, True)

submission.to_csv("submission.csv", index=False)



