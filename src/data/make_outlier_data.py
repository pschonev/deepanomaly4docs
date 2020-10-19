# %%
from itertools import islice
import pandas as pd

df_20 = pd.read_csv(
    "/home/philipp/projects/dad4td/data/external/20_newsgroup/20_newsgroup.csv")
df_imdb = pd.read_csv(
    "/home/philipp/projects/dad4td/data/external/imdb/IMDB Dataset.csv")

# %%

df_20 = df_20[["text", "target", "title"]]
df_20

# %%
df_imdb = df_imdb.rename(columns={'review': 'text'})
df_imdb = df_imdb[["text"]]
df_imdb["title"] = "imdb"
df_imdb["target"] = -1
df_imdb

# %%

df_comb = pd.concat([df_20, df_imdb], ignore_index=True)
df_comb

# %%
df_comb.info()

# %%
df_comb = df_comb.dropna()
df_comb["text"] = df_comb["text"].astype(str)
df_comb["title"] = df_comb["title"].astype(str)


df_comb["outlier_label"] = (df_comb["target"] * -1).clip(lower=0)
df_comb["outlier_label"][df_comb["outlier_label"] == 0] = -1
df_comb
# %%

# imdb has 50.000 entries, 20 newsgroup only 11.000
# 20 newsgroup is divied by categories, imdb does not have genre information (in this csv at least)

df_comb.to_pickle(
    "/home/philipp/projects/dad4td/data/processed/20_news_imdb.pkl")


# %%
# load the different datasets to be used as outliers and combine them into new dataframe
n_load = 10000
min_len = 100
max_len = 10000


def sf_to_df(filepath, n_load):
    df = pd.DataFrame(columns=["text"])
    with open(filepath, "r") as infile:
        for line in islice(infile, n_load):
            df = df.append({"text": line}, ignore_index=True)
    return df


df = pd.DataFrame(columns=["text"])
df_all_news = pd.read_csv(
    "/home/philipp/projects/dad4td/data/raw/all-the-news-2-1.csv", nrows=n_load)
df["text"] = df_all_news["article"]
df_amazon = pd.read_csv("/home/philipp/projects/dad4td/data/raw/amazon.csv", nrows=n_load, header=None,
                        usecols=[2], names=["text"])
df = df.append(df_amazon, ignore_index=True)
df_wiki = sf_to_df(
    "/home/philipp/projects/dad4td/data/processed/wiki_all_sf.txt", n_load=n_load)
df = df.append(df_wiki, ignore_index=True)
df_imdb = pd.read_csv(
    "/home/philipp/projects/dad4td/data/external/imdb/IMDB Dataset.csv", nrows=n_load)
df = df.append(df_imdb["review"].rename("text").to_frame(), ignore_index=True)

df = df[df["text"].str.len().between(min_len, max_len)]
df["text"].str.len().plot.hist(bins=50)

df.to_pickle("/home/philipp/projects/dad4td/data/processed/oe_data.pkl")
df

# %%
# get pagexml file content to dataframe
import pandas as pd
from pathlib import Path

label_file = [("test", "/home/philipp/projects/dad4td/data/raw/rvl_cdip/labels/test.txt"),
              ("train", "/home/philipp/projects/dad4td/data/raw/rvl_cdip/labels/train.txt"),
              ("val", "/home/philipp/projects/dad4td/data/raw/rvl_cdip/labels/val.txt")]
ocr_path = "/home/philipp/projects/dad4td/data/raw/rvl_cdip/data/"

df = pd.DataFrame(columns=["filename", "target", "split"])
for split, path in label_file:
    df_part = pd.read_csv(path, sep=" ", names=["filename", "target"])
    df_part["split"] = split
    df = df.append(df_part)

df["filename"] = df["filename"].map(lambda x: Path(x).stem)
df["isfile"] = df["filename"].map(lambda x: True if Path(f"{ocr_path}{x}_ocr.xml").is_file() else False)

df
# %%
df[df["isfile"] == True]["target"].value_counts()

#%%
label_df = "/home/philipp/projects/dad4td/data/raw/rvl_cdip/labels/labels.tsv"
df.to_csv(label_df, sep="\t")

#%%
# extract ocr text from xml to dataframe
import xml.etree.ElementTree as ET

def get_text(filepath):
    NSMAP = {'mw':'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        out_str = " ".join((x.text for x in root.findall('.//mw:Unicode', namespaces=NSMAP)))
        return out_str
    except FileNotFoundError:
        return ""

rvl_cdip_out = "/home/philipp/projects/dad4td/data/processed/rvl_cdip.pkl"

test_file = "/home/philipp/projects/dad4td/data/raw/rvl_cdip/data/0000036982_ocr.xml"

get_text(test_file)
df_text = df[df["isfile"] == True]
df_text = df_text.drop(["isfile"], axis=1).reset_index(drop=True)
df_text["text"] = df_text["filename"].map(lambda x: get_text(f"{ocr_path}{x}_ocr.xml"))
# %%
df_text.to_pickle(rvl_cdip_out)
df_text
