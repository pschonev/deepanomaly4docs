# %%
from keras.layers import Input, Dense, concatenate, Dropout
from keras.utils import plot_model
from keras.layers import Input, Dense, concatenate
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
from utils import remove_short_texts, get_scores, next_path, product_dict, umap_reduce
from dotmap import DotMap
import gc
import tensorflow as tf

tqdm.pandas(desc="progess: ")


# %%
rvl_cdip_img_embs = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg.pkl"
df = pd.read_pickle(rvl_cdip_img_embs)
df

# %%


def get_text(path):
    path = path[:-3]+"txt"
    text = open(path, "r").read()
    return text

#print("get text")
#df["text"] = df.filename.progress_map(lambda x: get_text(x))
# df
# %%
# df.to_pickle(rvl_cdip_img_embs)
# %%


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
vec_col = "vecs"
df = df.rename(columns={vec_col: "vecs"})
df.vecs = df.vecs.map(lambda x: x.flatten())

df
# %%
df = df.dropna()
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
df_t["doc2vec"] = doc2vecwikiall.vectorize(df_t["text"])
print("get train reference vecs")
df_r["doc2vec"] = doc2vecwikiall.vectorize(df_r["text"])
print("get test vecs")
df_test["doc2vec"] = doc2vecwikiall.vectorize(df_test["text"])

# %%
# one class


def create_model(n_in_img, n_in_text, dropout_rate=0.3):

    img_input = Input(shape=(n_in_img,), name='img_input')
    img = Dropout(dropout_rate)(img_input)
    img_output = Dense(n_in_text, activation='relu')(img)

    text_input = Input(shape=(n_in_text,), name='text_input')
    text = Dropout(dropout_rate)(text_input)
    text_output = Dense(n_in_text, activation='relu')(text)

    concat = concatenate([img_output, text_output], name='Concatenate')
    x = Dense(256, activation='relu')(concat)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    final_model = Model(inputs=[img_input, text_input], outputs=x,
                        name='Final_output')
    final_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return final_model


def create_sup_model(n_in, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(n_in, input_dim=n_in, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(n_in/2), activation='relu'))
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




def prep_data(chosen_class, df_t, df_r, df_test, c, random_state=42, ref_data = "same"):

    if ref_data == "same":
        if c.weakly:
            chosen_samples = df_r.where(df_r.target.isin(chosen_class)).dropna().sample(n=c.weakly, random_state=random_state)

        df_r = df_r.where(~df_r.target.isin(chosen_class)).dropna()
        if c.weakly:
            df_r = df_r.append(chosen_samples)
    print(f"df targets:\n{df_t.target.value_counts()}")
    print(f"df references:\n{df_r.target.value_counts()}")

    remap = {k: v for v, k in zip(
        range(df_r.target.unique().shape[0]), df_r.target.unique())}
    df_r.target = df_r.target.map(remap)

    print(f"df references:\n{df_r.target.value_counts()}")

    if c.balanced:
        df_test_out = df_test.where(
            (df_test.label == 0) & (df_test.target.isin(chosen_class))).dropna()
        n=df_test_out.shape[0]
        df_test_in = df_test.where(df_test.label == 1)
        df_test_in = df_test_in.dropna(how="all").sample(
            n=n, random_state=random_state)
        df_test = df_test_out.append(df_test_in).dropna(how="all")
    else:
        df_test_out = df_test.where(df_test.target.isin(chosen_class)).dropna()
        df_test_in = df_test.where(df_test.target.isin(inliers)).dropna(
                                            ).sample(n=df_test_out.shape[0]*4, random_state=random_state)
        df_test = pd.concat([df_test_in, df_test_out])
    print(f"df test:\n{df_test.target.value_counts()}")

    return df_t, df_r, df_test


def train_test(result_df, df_t, df_r, df_test, res_path, c):

    # data
    y_ref = np.array(df_r.target.to_list())
    

    
    x_target = np.array(df_t.vecs.to_list())
    x_target, img_umap_model = umap_reduce(x_target, c.use_umap, logstr="x_target image")
    x_ref = np.array(df_r.vecs.to_list())
    x_ref, _ = umap_reduce(x_ref, c.use_umap, umap_model=img_umap_model, logstr="x_ref image")

    x_text_target = np.array(df_t.doc2vec.to_list())
    x_text_target, txt_umap_model = umap_reduce(x_text_target, c.use_umap, logstr="x_target text")
    x_text_ref = np.array(df_r.doc2vec.to_list())
    x_text_ref, _ = umap_reduce(x_text_ref, c.use_umap, umap_model=txt_umap_model, logstr="x_ref text")
    
    y_ref = to_categorical(y_ref)

    test_vecs = np.array(df_test.vecs.to_list())
    test_vecs, _ = umap_reduce(test_vecs, c.use_umap, umap_model=img_umap_model, logstr="test_vecs image")
    test_text_vecs = np.array(df_test.doc2vec.to_list())
    test_text_vecs, _ = umap_reduce(test_text_vecs, c.use_umap, umap_model=txt_umap_model, logstr="test_vecs text")

    n_sup = 10000
    n_per_targ = 1000
    df_r_temp = df_r.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_per_targ), random_state=42))

    x_tr = np.array(df_t.head(n_sup).append(df_r_temp).vecs.to_list())
    x_tr, _ = umap_reduce(x_tr, c.use_umap, umap_model=img_umap_model, logstr="x_tr image")
    x_t_tr = np.array(df_t.head(n_sup).append(df_r_temp).doc2vec.to_list())
    x_t_tr, _ = umap_reduce(x_t_tr, c.use_umap, umap_model=txt_umap_model, logstr="x_tr text")
    y_tr = np.array(df_t.head(n_sup).append(df_r_temp).label.to_list())

    #y_tr = to_categorical(y_tr)

    #print(f"{df.where(df.label == 0).dropna().target.value_counts()}")

    #print(f"x_target: {x_target.shape}\nx_ref: {x_ref.shape}\ny_ref: {y_ref.shape}\n")

    classes = df_r.target.unique().shape[0]
    print(f"classes: {classes}")
    batchsize = 128
    epoch_num = 30
    epoch_report = 4
    feature_out = 64
    pred_mode = "nn"

    # get the loss for compactness
    original_loss = create_loss(classes, batchsize)

    # model creation
    model = create_model(
        n_in_img=x_target[0].shape[0], n_in_text=x_text_target[0].shape[0])

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

    for epochnumber in range(c.epoch_num):
        x_r, x_t_r, y_r, lc, ld = [], [], [], [], []

        np.random.shuffle(x_target)

        np.random.shuffle(ref_samples)
        for i in range(len(x_ref)):
            x_r.append(x_ref[ref_samples[i]])
            x_t_r.append(x_text_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        x_t_r = np.array(x_t_r)
        y_r = np.array(y_r)

        for i in range(int(len(x_target) / batchsize)):
            batch_target = x_target[i*batchsize:i*batchsize+batchsize]
            batch_text_target = x_text_target[i *
                                              batchsize:i*batchsize+batchsize]
            batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
            batch_text_ref = x_t_r[i*batchsize:i*batchsize+batchsize]
            batch_y = y_r[i*batchsize:i*batchsize+batchsize]
            # target data
            lc.append(model_t.train_on_batch([batch_target, batch_text_target],
                                             np.zeros((batchsize, feature_out))))

            # reference data
            ld.append(model_r.train_on_batch(
                [batch_ref, batch_text_ref], batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))
        epochs.append(epochnumber)

        if epochnumber % c.epoch_report == 0 or epochnumber == c.epoch_num-1:
            print(
                f"-----\n\nepoch : {epochnumber+1} ,Descriptive loss : {loss[-1]}, Compact loss : {loss_c[-1]}")

            #test_b = model_t.predict(test_vecs)

            #od = OCSVM()
            # od.fit(test_b)

            #decision_scores = od.labels_

            # decision_scores = decision_scores.astype(float)

            labels = df_test["label"].astype(int).values

            # threshold = 0.5
            # scores = get_scores(dict(),labels, np.where(decision_scores > threshold, 0, 1), outlabel=0)
            if c.pred_mode == "svm":
                x_tr_pred = model_t.predict(x_tr)
                clf = SVC(probability=True)
                clf.fit(x_tr_pred, y_tr)

                preds = model_t.predict(test_vecs)
                decision_scores = clf.predict_proba(preds)
                decision_scores = np.array([x[1] for x in decision_scores])
            elif c.pred_mode == "ocsvm":
                x_tr_pred = model_t.predict(x_tr)
                clf = OneClassSVM()
                clf.fit(x_tr_pred)

                preds = model_t.predict(test_vecs)
                decision_scores = clf.score_samples(preds)
            elif c.pred_mode == "nn":
                y_tr = y_tr.astype(int)
                print(y_tr)
                x_tr_pred = model_t.predict([x_tr, x_t_tr])
                clf = create_sup_model(n_in=c.feature_out)
                clf.summary()
                clf.fit(
                    x_tr_pred, y=y_tr, epochs=c.sup_epochs, batch_size=64, verbose=True)

                decision_scores = model_t.predict([test_vecs, test_text_vecs])
                decision_scores = clf.predict(decision_scores)
                decision_scores = decision_scores.astype(float)
#                _ = plt.hist(preds, bins=10)
#                plt.show()

                #cleanup supervised model
                del clf
                gc.collect()
                K.clear_session()
                tf.compat.v1.reset_default_graph()
                

            else:
                raise Exception(f"{c.pred_mode} must be one of svm, nn, ocsvm")
            
            scores = get_scores(labels, decision_scores, outlabel=0, threshold=c.threshold)
            print(f"\n\nTest scores:\n{pd.DataFrame([scores], index=[0])}")
            if scores["pr_auc"] > best_acc and epochnumber != 0:
                best_acc = scores["pr_auc"]
                best_scores = scores
                scores["fpr"], scores["tpr"], _= metrics.roc_curve(labels, decision_scores)
                scores["precision"], scores["recall"], _ = metrics.precision_recall_curve(labels, decision_scores)
                print(f"best_acc updated to: {best_acc}")

    result_df = result_df.append(dict(cclass=list(
        df_test.target.unique()), **best_scores, **c), ignore_index=True)
    result_df.to_csv(res_path, sep="\t")
    result_df.to_pickle(res_path[:-3]+"pkl")

    #cleanup models
    del model_r
    del model_t
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    return result_df
    

# %%
    
# config
mode = "one_out"
c = DotMap()
c.weakly = [None]
c.batchsize = [128]
c.epoch_num = [12]
c.epoch_report = [4]
c.sup_epochs = [15]
c.feature_out = [64]
c.pred_mode = ["nn"]
c.threshold = [0.55]
c.n_sup = [10000] # samples per inlier class for final fcnn
c.n_per_targ = [1000] # samples per outlier class (reference data) for fcnn
c.random_state = range(1,5)
c.balanced = [False]
c.use_umap = [False]

c = [DotMap(x) for x in product_dict(**c)]

i=0
result_df = pd.DataFrame()
if mode == "one_out":
    res_path = next_path(
        "/home/philipp/projects/dad4td/reports/one_class/one_out_oc_%04d_nn_combined_prc.tsv")
    for i in outliers:
        print(f"class in test: {i}")
        for params in c:
            print(params)
            result_df = train_test(result_df, *prep_data([i], df_t, df_r, df_test, c=params, ref_data =ref_data), 
                res_path=res_path, c=params)
elif mode == "all":
    res_path = next_path(
        "/home/philipp/projects/dad4td/reports/one_class/all_oc%04d_all_bub.tsv")
    remap = {k: v for v, k in zip(
        range(df_r.target.unique().shape[0]), df_r.target.unique())}
    df_r_temp = df_r.copy(deep=True)
    df_r_temp.target = df_r_temp.target.map(remap)
    for params in c:
        print(params)
        result_df = train_test(result_df, df_t, df_r_temp, df_test, res_path=res_path, c=params)
elif mode == "single_one_out":
    res_path = next_path(
        "/home/philipp/projects/dad4td/reports/one_class/single_one_out_%04d_20news.tsv")
    i = 6
    for params in c:
        print(params)
        result_df = train_test(result_df, *prep_data([i], df_t, df_r, df_test, c=params, ref_data =ref_data), 
            res_path=res_path, c=params)

#%%