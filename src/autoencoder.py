# %%
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


def remove_short_texts(df, min_len=5):
    n_before = df.shape[0]
    df = df[df['text'].map(len) > min_len]
    print(
        f"Removed {n_before - df.shape[0]} rows with doc length below {min_len}.")
    return df


data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
df = pd.read_pickle(data_path)
df = remove_short_texts(df)


num_epochs = 100
batch_size = 256
learning_rate = 1e-3
dropout_rate = 0.2

#data = list(df["apnews_256"].values)
text = np.vstack(df["apnews_256"].to_numpy())
labels = df["outlier_label"].to_numpy()
#%%
text = torch.Tensor(text)
labels = torch.Tensor(labels)
data = TensorDataset(text, labels)

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
next(iter(dataloader))
#%%

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Linear(256, 128),
            # nn.ReLU(True), 
            # nn.Dropout(dropout_rate),
            # nn.Linear(128, 64),
            # nn.ReLU(True), 
            # nn.Dropout(dropout_rate),
            # nn.Linear(64, 12),
            nn.Linear(256, 1),
            )
        self.decoder = nn.Sequential(
            # nn.Linear(12, 64),
            # nn.ReLU(True),
            # nn.Dropout(dropout_rate),
            # nn.Linear(64, 128),
            # nn.ReLU(True), 
            # nn.Dropout(dropout_rate), 
            # nn.Linear(128, 256), 
            nn.Linear(1, 256),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for i, data_labels in enumerate(dataloader):
        data = data_labels[0]
        labels = data_labels[1]
        model.train()
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        if i <= 2:
            print(f'loss:{loss.data}')
            model.eval()
            with torch.no_grad():
                for ten in data:
                    print(criterion(model(ten), ten))

    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.data}')
    if epoch % 2 == 0:
        print("Test - ", epoch)
