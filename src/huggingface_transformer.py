#%%
import pandas as pd
from evaluation import Doc2VecModel
from tqdm import tqdm

tqdm.pandas(desc="progess: ")
# %%
df = pd.read_pickle("/home/philipp/projects/dad4td/data/raw/QS-OCR-Large/rvl_cdip.pkl")
inliers = [0, 1, 2, 11]
unused_classes = [3, 8]
n_class = 500
contamination = 0.1

df = df.where(~df.target.isin(unused_classes))
df["label"] = 0
df.loc[df.target.isin(inliers), "label"] = 1

# get only n samples
df = df.groupby('target', group_keys=False).apply(
            lambda df: df.sample(n=min(df.shape[0], n_class), random_state=42))
# shuffle
df = df.sample(frac=1)
# apply contamination factor
x_n = df[df.label == 1].shape[0]
df = df[df["label"] == 1].head(x_n).append(
    df[df["label"] == 0].head(int(x_n*contamination)))

df = df.reset_index(drop=True)
df.target.value_counts()

df_full = df.copy(deep=True)
# %%

doc2vecwikiall = Doc2VecModel("doc2vec_wiki_all", "wiki_EN", 1.0,
                              100, 1, "/home/philipp/projects/dad4td/models/enwiki_dbow/doc2vec.bin")
df["vecs"] = doc2vecwikiall.vectorize(df["text"])

# %%
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
granola_ids = tokenizer.encode('granola bars')

# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))
print('granola_tokens', tokenizer.convert_ids_to_tokens(granola_ids))

# Convert the list of IDs to a tensor of IDs 
granola_ids = torch.LongTensor(granola_ids)
# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))

config = DistilBertConfig.from_pretrained(model_name, output_hidden_states=True)
model = DistilBertModel.from_pretrained(model_name, config=config)
# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
granola_ids = granola_ids.to(device)

model.eval()

print(granola_ids.size())
# unsqueeze IDs to get batch size of 1 as added dimension
granola_ids = granola_ids.unsqueeze(0)
print(granola_ids.size())

print(type(granola_ids))
with torch.no_grad():
    out = model(input_ids=granola_ids)

# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[1]
print(len(hidden_states))

sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
print(sentence_embedding)
print(sentence_embedding.size())

# %%
min_len = 250
max_len = 4096
df = df_full
bef = df.shape[0]
df["text_len"] = df.text.map(lambda x: len(x))
df = df.where(df.text_len.between(min_len, max_len-1)).dropna().reset_index(drop=True)
print(
        f"Removed {bef-df.shape[0]} because they were under {min_len} or over {max_len} characters long.")
print(df.target.value_counts())


import torch
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
from transformers import LongformerTokenizerFast, LongformerModel, LongformerConfig

#model_name = 'distilbert-base-uncased'
model_name = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizerFast.from_pretrained(model_name)

df["vecs"] = df.text.map(lambda x: torch.LongTensor(tokenizer.encode(x)).unsqueeze(0))

config = LongformerConfig.from_pretrained(model_name, output_hidden_states=True)
model = LongformerModel.from_pretrained(model_name, config=config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model = model.to(device)
input_tf = tokenizer.batch_encode_plus(
        df.text.to_list(),
        return_tensors='pt',
        padding=True
    )
#vecs = input_tf['input_ids'].to(device)
#granola_ids = granola_ids.to(device)

model.eval()

with torch.no_grad():
    print("and GO!!!!")
    out = [model(input_ids=vec.to(device)) for vec in tqdm(df.vecs.to_list())]
with torch.no_grad():
    #df["vecs"] = df.vecs.progress_apply(lambda x: )
    #out = model(input_ids=vecs)
    print("what is happening here???")

# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[1]
print(len(hidden_states))