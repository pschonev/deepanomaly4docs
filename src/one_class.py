# %%
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from evaluation import Doc2VecModel
from tqdm import tqdm
import numpy as np
from evaluation import get_scores
from utils import remove_short_texts

tqdm.pandas(desc="progess: ")


# %%
df = pd.read_pickle(
    "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl")
inliers = [0, 1, 2, 11]
outliers = [4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
unused_classes = [3, 8]
n_class = 500
contamination = 0.1
min_len = 250
ref_data = "same"

df = df.where(~df.target.isin(unused_classes))
df["label"] = 0
df.loc[df.target.isin(inliers), "label"] = 1
# %%
df = df.dropna()
df = remove_short_texts(
                    df=df, data_name="Full DF", min_len=250)
# %%
# get only n samples
df = df.groupby('target', group_keys=False).apply(
    lambda df: df.sample(n=min(df.shape[0], n_class), random_state=42))
df = df.reset_index(drop=True)
df.target.value_counts()

# %%
# shuffle
df = df.sample(frac=1)
# apply contamination factor
x_n = df[df.label == 1].shape[0]
df = df[df["label"] == 1].head(x_n).append(
    df[df["label"] == 0].head(int(x_n*contamination)))

df = df.reset_index(drop=True)
df.target.value_counts()

# %%
# split data into train_target, train_reference and test
df, df_test = train_test_split(df, test_size=int(df.shape[0]*0.1), random_state=42,
                               stratify=df["target"])
df_t = df.where(df.label == 1).dropna()
if ref_data == "same":
    df_r = df.where(df.label == 0).dropna()
elif ref_data == "20_news":
    df_r = pd.read_pickle("/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_r = df_r.where(df_r.target != -1).dropna()
elif ref_data == "both":
    df_r = df.where(df.label == 0).dropna()
    df_20 = pd.read_pickle("/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_20 = df_20.where(df_20.target != -1).dropna()
    df_20.target = (df_20.target + 1)*20
    df_r = df_r.append(df_20)
else:
    raise Exception(f"{ref_data} not valid value for ref_data. Must be one of: same, 20_news")
df_r
# %%
df_r.target.value_counts()
# %%
remap = {k:v for v,k in zip(range(df_r.target.unique().shape[0]), df_r.target.unique())}
df_r.target =  df_r.target.map(remap)
# %%
print("df_t\n",df_t.target.value_counts())
print("df_r\n", df_r.target.value_counts())
print("df_test\n", df_test.target.value_counts())
# %%

doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
print("get train target vecs")
df_t["vecs"] = doc2vecwikiall.vectorize(df_t["text"])
print("get train reference vecs")
df_r["vecs"] = doc2vecwikiall.vectorize(df_r["text"])
print("get test vecs")
df_test["vecs"] = doc2vecwikiall.vectorize(df_test["text"])

# %%
# one class
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.engine.network import Network
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from pyod.models.ocsvm import OCSVM

def create_model(loss="categorical_crossentropy", dropout_rate=0.2, n_in=256):
    model = Sequential()
    model.add(Dense(256, input_dim=n_in, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))

    model.compile(loss=loss,
                  optimizer='adam', metrics=['accuracy'])
    return model

def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]   ) / ((batchsize-1)**2)
    return lc

# data

x_target = np.array(df_t.vecs.to_list())
x_ref = np.array(df_r.vecs.to_list())
y_ref = np.array(df_r.target.to_list())
y_ref = to_categorical(y_ref)
test_vecs = np.array(df_test.vecs.to_list())

#print(f"{df.where(df.label == 0).dropna().target.value_counts()}")

#print(f"x_target: {x_target.shape}\nx_ref: {x_ref.shape}\ny_ref: {y_ref.shape}\n")

classes = df_r.target.unique().shape[0]
print(f"classes: {classes}")
batchsize = 128
epoch_num = 300
feature_out = 512

# model creation
model = create_model(loss="binary_crossentropy",
                     n_in=x_target[0].shape[0])

model_t = Model(inputs=model.input,outputs=model.output)

model_r = Network(inputs=model_t.input,
                    outputs=model_t.output,
                    name="shared_layer")

prediction = Dense(classes, activation='softmax')(model_t.output)
model_r = Model(inputs=model_r.input,outputs=prediction)

prediction_t = Dense(feature_out, activation='softmax')(model_t.output)
model_t = Model(inputs=model_t.input,outputs=prediction_t)

#optimizer = SGD(lr=5e-5, decay=0.00005)
optimizer = Adam(learning_rate=5e-5)

model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
model_t.compile(optimizer=optimizer, loss=original_loss)

model_t.summary()
model_r.summary()

ref_samples = np.arange(x_ref.shape[0])
loss, loss_c = [], []
epochs = []
print("training...")

for epochnumber in range(epoch_num):
    x_r, y_r, lc, ld = [], [], [], []

    np.random.shuffle(x_target)

    np.random.shuffle(ref_samples)
    for i in range(len(x_ref)):
        x_r.append(x_ref[ref_samples[i]])
        y_r.append(y_ref[ref_samples[i]])
    x_r = np.array(x_r)
    y_r = np.array(y_r)

    for i in range(int(len(x_target) / batchsize)):
        batch_target = x_target[i*batchsize:i*batchsize+batchsize]
        batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
        batch_y = y_r[i*batchsize:i*batchsize+batchsize]
        #target data
        lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

        #reference data
        ld.append(model_r.train_on_batch(batch_ref,batch_y))

    loss.append(np.mean(ld))
    loss_c.append(np.mean(lc))
    epochs.append(epochnumber)

    
    if epochnumber % 5 == 0:
        print("epoch : {} ,Descriptive loss : {}, Compact loss : {}".format(epochnumber+1, loss[-1], loss_c[-1]))

        model_t.save_weights('/home/philipp/projects/dad4td/models/one_class/model_t_smd_{}.h5'.format(epochnumber))
        model_r.save_weights('/home/philipp/projects/dad4td/models/one_class/model_r_smd_{}.h5'.format(epochnumber))
        test_b = model_t.predict(test_vecs)

        #test_b = test_b.reshape((len(test_b),-1))
        #ms = MinMaxScaler()
        #test_b = ms.fit_transform(test_b)

        od = OCSVM()
        od.fit(test_b)

        decision_scores = od.labels_
        
        decision_scores = decision_scores.astype(float)

        labels = df_test["label"].astype(int).values

        threshold = 0.5
        scores = get_scores(dict(),labels, np.where(decision_scores > threshold, 0, 1), outlabel=0)
        print(f"\nTest scores:\n{pd.DataFrame([scores], index=[0])}")

# %%
