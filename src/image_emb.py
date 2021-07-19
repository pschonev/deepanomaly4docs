# %%

import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from tqdm import tqdm

tqdm.pandas(desc="progess: ")

# %%
label_file_path = "/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/labels.tsv"
base_path = "/media/philipp/Fotos/rvl-cdip/"
df = pd.read_csv(label_file_path, sep="\t")
df = df[["filename", "target", "split"]]
df.filename = df.filename.map(lambda x: base_path + x[:-4] + ".tif")
df

# %%
# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)

def get_imgvec(path):
    # Read in an image
    img = Image.open('/media/philipp/Fotos/rvl-cdip/images/imagesa/a/a/a/aaa06d00/50486482-6482.tif')
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg
    # Get a vector from img2vec, returned as a torch FloatTensor
    vec = img2vec.get_vec(rgbimg, tensor=True)
    return vec

# %%
df["resnet18_emb"] = df.filename.progress_map(lambda x: get_imgvec(x)).to_list()

# %%
df.to_pickle(base_path+"rvl_cdip_embs.pkl")

# %%
df.resnet18_emb = df.resnet18_emb.progress_map(lambda x: x.detach().numpy().flatten())
# %%

n_per_targ = 100
df_sample = df.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_per_targ), random_state=42))
df_sample

# -------------- VG 16 --------------------

# %%
from keras.layers import Input, Dense
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd 
from tqdm import tqdm

tqdm.pandas(desc="progess: ")
# %%

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# %% get data
data_file_path = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_embs.pkl"
df = pd.read_pickle(data_file_path)
df

# %% sample
n_per_targ = 12500
unused_classes = [3, 8]

df_sample = df.where(~df.target.isin(unused_classes))
df_sample = df_sample.groupby('target', group_keys=False).apply(
        lambda df: df.sample(n=min(df.shape[0], n_per_targ), random_state=42))
df_sample

# %% create model

vgg16 = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=vgg16.input, outputs=vgg16.layers[-3].output)
model.summary()

# %%
img_path = '/media/philipp/Fotos/rvl-cdip/images/imagesa/a/a/a/aaa06d00/50486482-6482.tif'
x = preprocess_img(img_path)
features = model.predict(x)
features.shape

# %%
df_sample["vecs"] = df_sample.filename.progress_map(lambda x: model.predict(preprocess_img(x)))
df_sample.vecs

# %%
out_path = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg.pkl"

df_sample[["filename", "target", "vecs"]].to_pickle(out_path)

# %%
from umap import UMAP
import pandas as pd 

out_path = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg.pkl"

df = pd.read_pickle(out_path)
df

# %%
vecs = df.vecs.to_list()
vecs = [x.flatten() for x in vecs]

red_vecs = UMAP(verbose=True, n_components=300).fit_transform(vecs)

# %%
df.vecs.shape

# %%
df["vecs_300"] = red_vecs
df.to_pickle(out_path) 