# %%
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.engine.network import Network
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from evaluation import Doc2VecModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import remove_short_texts, get_scores

tqdm.pandas(desc="progess: ")


# %%
df = pd.read_pickle(
    "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl")
inliers = [0, 1, 2, 11]
outliers = [4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
unused_classes = [3, 8]
n_class = 20000
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
    df_r = pd.read_pickle(
        "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_r = df_r.where(df_r.target != -1).dropna()
elif ref_data == "both":
    df_r = df.where(df.label == 0).dropna()
    df_20 = pd.read_pickle(
        "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")
    df_20 = df_20.where(df_20.target != -1).dropna()
    df_20.target = (df_20.target + 1)*20
    df_r = df_r.append(df_20)
else:
    raise Exception(
        f"{ref_data} not valid value for ref_data. Must be one of: same, 20_news")
df_r
# %%
df_r.target.value_counts()
# %%
remap = {k: v for v, k in zip(
    range(df_r.target.unique().shape[0]), df_r.target.unique())}
df_r.target = df_r.target.map(remap)
# %%
print("df_t\n", df_t.target.value_counts())
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


def create_model(loss="categorical_crossentropy", dropout_rate=0.2, n_in=256):
    model = Sequential()
    model.add(Dense(128, input_dim=n_in, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(dropout_rate))
    #model.add(Dense(128, activation='relu'))
    # model.add(Dropout(dropout_rate))
    #model.add(Dense(128, activation='relu'))
    # model.add(Dropout(dropout_rate))
    #model.add(Dense(64, activation='relu'))

    model.compile(loss=loss,
                  optimizer='adam', metrics=['accuracy'])
    return model


def create_sup_model(n_in, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(4, input_dim=n_in, activation='relu'))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(int(n_in/2), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=['accuracy'])
    return model


def create_loss(classes, batchsize):
    def original_loss(y_true, y_pred):
        lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -
                                                           K.mean(y_pred, axis=0))**2, axis=[1]) / ((batchsize-1)**2)
        return lc
    return original_loss


def prep_data(chosen_class, df_t, df_r, df_test):
    df_r = df_r.where(~df_r.target.isin(chosen_class)).dropna()
    print(f"df targets:\n{df_t.target.value_counts()}")
    print(f"df references:\n{df_r.target.value_counts()}")

    remap = {k: v for v, k in zip(
        range(df_r.target.unique().shape[0]), df_r.target.unique())}
    df_r.target = df_r.target.map(remap)

    print(f"df references:\n{df_r.target.value_counts()}")

    df_test = df_test.where(df_test.target.isin(chosen_class)).dropna()
    print(f"df test:\n{df_test.target.value_counts()}")

    return df_t, df_r, df_test


def train_test(result_df, df_t, df_r, df_test):

    # data

    x_target = np.array(df_t.vecs.to_list())
    x_ref = np.array(df_r.vecs.to_list())
    y_ref = np.array(df_r.target.to_list())
    y_ref = to_categorical(y_ref)
    test_vecs = np.array(df_test.vecs.to_list())

    n_sup = 10000
    n_per_targ = 1000
    df_r_temp = df_r.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_per_targ), random_state=42))

    x_tr = np.array(df_t.head(n_sup).append(df_r_temp).vecs.to_list())
    y_tr = np.array(df_t.head(n_sup).append(df_r_temp).label.to_list())

    #y_tr = to_categorical(y_tr)

    #print(f"{df.where(df.label == 0).dropna().target.value_counts()}")

    #print(f"x_target: {x_target.shape}\nx_ref: {x_ref.shape}\ny_ref: {y_ref.shape}\n")

    res_path = "/home/philipp/projects/dad4td/reports/one_class/all.tsv"
    classes = df_r.target.unique().shape[0]
    print(f"classes: {classes}")
    batchsize = 128
    epoch_num = 15
    epoch_report = 5
    feature_out = 64
    pred_mode = "nn"

    # get the loss for compactness
    original_loss = create_loss(classes, batchsize)

    # model creation
    model = create_model(loss="binary_crossentropy",
                         n_in=x_target[0].shape[0])

    model_t = Model(inputs=model.input, outputs=model.output)

    model_r = Network(inputs=model_t.input,
                      outputs=model_t.output,
                      name="shared_layer")

    prediction = Dense(classes, activation='softmax')(model_t.output)
    model_r = Model(inputs=model_r.input, outputs=prediction)

    #latent_t = Dense(2, activation='relu')(model_t.output)
    #model_t = Model(inputs=model_t.input,outputs=latent_t)
    prediction_t = Dense(feature_out, activation='softmax')(model_t.output)
    model_t = Model(inputs=model_t.input, outputs=prediction_t)

    #optimizer = SGD(lr=5e-5, decay=0.00005)
    optimizer = Adam(learning_rate=5e-5)

    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    model_t.summary()
    model_r.summary()

    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []
    epochs = []
    best_acc = 0
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
            # target data
            lc.append(model_t.train_on_batch(batch_target,
                                             np.zeros((batchsize, feature_out))))

            # reference data
            ld.append(model_r.train_on_batch(batch_ref, batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))
        epochs.append(epochnumber)

        if epochnumber % epoch_report == 0 or epochnumber == epoch_num-1:
            print(
                f"-----\n\nepoch : {epochnumber+1} ,Descriptive loss : {loss[-1]}, Compact loss : {loss_c[-1]}")

            model_t.save_weights(
                '/home/philipp/projects/dad4td/models/one_class/model_t_smd_{}.h5'.format(epochnumber))
            model_r.save_weights(
                '/home/philipp/projects/dad4td/models/one_class/model_r_smd_{}.h5'.format(epochnumber))
            #test_b = model_t.predict(test_vecs)

            #od = OCSVM()
            # od.fit(test_b)

            #decision_scores = od.labels_

            # decision_scores = decision_scores.astype(float)

            labels = df_test["label"].astype(int).values

            # threshold = 0.5
            # scores = get_scores(dict(),labels, np.where(decision_scores > threshold, 0, 1), outlabel=0)
            if pred_mode == "svm":
                x_tr_pred = model_t.predict(x_tr)
                clf = SVC()
                clf.fit(x_tr_pred, y_tr)

                preds = model_t.predict(test_vecs)
                preds = clf.predict(preds)
            elif pred_mode == "nn":
                y_tr = y_tr.astype(int)
                print(y_tr)
                x_tr_pred = model_t.predict(x_tr)
                clf = create_sup_model(n_in=feature_out)
                clf.summary()
                clf.fit(
                    x_tr_pred, y=y_tr, epochs=15, batch_size=64, verbose=True)

                decision_scores = model_t.predict(test_vecs)
                decision_scores = clf.predict(decision_scores)
                preds = decision_scores.astype(float)
                
                _ = plt.hist(preds, bins=10)
                plt.show()
                

            else:
                raise Exception(f"{pred_mode} must be one of svm, nn, osvm")
            
            scores = get_scores(dict(), labels, preds, outlabel=0)
            print(f"\n\nTest scores:\n{pd.DataFrame([scores], index=[0])}")
            if scores["accuracy"] > best_acc:
                best_acc = scores["accuracy"]
                print(f"best_acc updated to: {best_acc}")
            normalize="true"
            print(f"{confusion_matrix(labels, preds, normalize=normalize)}")
    result_df = result_df.append(dict(cclass=list(
        df_test.target.unique()), accuracy=best_acc), ignore_index=True)
    result_df.to_csv(res_path, sep="\t")
    return result_df


result_df = pd.DataFrame()
mode = "all"

if mode == "one_out":
    for i in inliers:
        print(f"class in test: {i}")
        result_df = train_test(result_df, *prep_data([i], df_t, df_r, df_test))
elif mode == "all":
    print(df_t.target.value_counts(), df_r.target.value_counts(),
          df_test.target.value_counts())
    train_test(result_df, df_t, df_r, df_test)
# %%
