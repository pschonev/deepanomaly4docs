# %%
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


def remove_short_texts(df, min_len=5):
    n_before = df.shape[0]
    df = df[df['text'].map(len) > min_len]
    print(
        f"Removed {n_before - df.shape[0]} rows with doc length below {min_len}.")
    return df


def sample_data(df, fraction, contamination, seed=42):
    X_n = int(df.shape[0] * fraction)
    y_n = int(X_n * contamination)

    df = df.iloc[np.random.RandomState(seed=seed).permutation(len(df))]
    df = df[df["outlier_label"] == 1].head(X_n).append(
        df[df["outlier_label"] == -1].head(y_n))
    df = df.reset_index(drop=True)
    return df


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(sr, [pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return ((np.abs(sr - median)) >= iqr/2)


def get_f1(scores, labels, iq_range=0.5):
    preds = reject_outliers(scores, iq_range=1.0-contamination)
    preds = [-1 if x else 1 for x in preds]
    f1_macro = f1_score(labels, preds, average='macro')
    in_f1 = f1_score(labels, preds, pos_label=1)
    out_f1 = f1_score(labels, preds, pos_label=-1)
    
    return f1_macro, in_f1, out_f1


data_path = "/home/philipp/projects/dad4td/data/processed/20_news_imdb_vec.pkl"
vec_col = "doc2vecwikiimdb20news013030_256"
log_path = "/home/philipp/projects/dad4td/reports/logs/doc2vecwikiimdb20news013030_256-8"
df = pd.read_pickle(data_path)
df = remove_short_texts(df)

num_epochs = 10000
batch_size = 64
learning_rate = 1e-3
dropout_rate = 0.4
log_interval = 5

#val data
fraction = 0.15
contamination = 0.1

# train data
text = np.vstack(df[vec_col].to_numpy())
labels = df["outlier_label"].to_numpy()
text = torch.Tensor(text)
labels = torch.Tensor(labels)
data = TensorDataset(text, labels)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# val data
df_val = sample_data(df, fraction, contamination)
text_val = np.vstack(df_val[vec_col].to_numpy())
labels_val = df_val["outlier_label"].to_numpy()
text_val = torch.Tensor(text_val)
labels_val = torch.Tensor(labels_val)
data_val = TensorDataset(text_val, labels_val)
dataloader_val = DataLoader(
    data_val, batch_size=df_val.shape[0], shuffle=False)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
writer = SummaryWriter(log_path)

running_loss = 0.0
losses = []
for epoch in range(num_epochs):
    for i, data_labels in enumerate(dataloader):
        data, labels = data_labels
        model.train()
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        running_loss += loss.item()
        if i % log_interval == 0 or epoch == 0:
            step = epoch * len(dataloader) + i
            writer.add_scalar('training loss', running_loss / log_interval, step)
            model.eval()
            with torch.no_grad():
                for val_data in dataloader_val:
                    val_embs, val_labels = val_data
                    for emb in val_embs:
                        losses.append(criterion(model(emb), emb))
                f1_macro, in_f1, out_f1 = get_f1(losses, val_labels)
                writer.add_scalar('f1_macro', f1_macro, step)
                writer.add_scalar('in_f1', in_f1, step)
                writer.add_scalar('out_f1', out_f1, step)
                writer.add_scalar('validation loss', sum(losses)/len(losses), step)
            running_loss = 0.0
            losses = []

    if epoch % 1 == 0:
        print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.data}')
