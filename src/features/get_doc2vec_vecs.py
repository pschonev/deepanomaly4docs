
# %%
import pandas as pd 
from embedders import Doc2VecModel
from tqdm import tqdm

tqdm.pandas(desc="progess: ")
# %%
rvl_cdip_img_embs = "/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg.pkl"
df = pd.read_pickle(rvl_cdip_img_embs)
df
# %%
doc2vecwikiall = Doc2VecModel(
                    model_name="doc2vec_wiki_all",
                    model_train_data="wiki_EN",
                    doc2vec_data_frac=1.0,
                    doc2vec_epochs=100,
                    doc2vec_min_count= 1,
                    model_path= "../models/enwiki_dbow/doc2vec.bin")
# %%
print("get train target vecs")
df["doc2vec"] = doc2vecwikiall.vectorize(X=df, data_col="text")

# %%
df.to_pickle("/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg_doc2vec.pkl")