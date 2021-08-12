from omegaconf import DictConfig
from src.supervised.supervised_eval import grid_search
from src.supervised.supervised import *
from src.embedders import PreComputed, Doc2VecModel
import hydra
from tqdm import tqdm

tqdm.pandas(desc="progess: ")

if __name__ == "__main__":
    eval_run = SupEvalRun(
        eval_mode=EvalMode.ONEOUT,
        data=DataHandler(
            database=Database(
                path="/media/philipp/Fotos/rvl-cdip/rvl_cdip_vgg_doc2vec.pkl"),
            txt_vecs=TextInput(
                name="doc2vec_wiki",
                data_col="doc2vec",
                vec_col="doc2vec",
                # embedder=Doc2VecModel(
                #     model_name="doc2vec_wiki_all",
                #     model_train_data="wiki_EN",
                #     doc2vec_data_frac=1.0,
                #     doc2vec_epochs=100,
                #     doc2vec_min_count= 1,
                #     model_path= "models/enwiki_dbow/doc2vec.bin")
                embedder=PreComputed(
                    model_name="Doc2Vec",
                    model_train_data="WikipediaEN"
                )
            ),
            img_vecs=ImageInput(
                name="vgg16",
                data_col="vecs_300",
                vec_col="vecs_300",
                embedder=PreComputed(
                    model_name="VGG16",
                    model_train_data="ImageNet"
                )
            ),
            labels=Labels(),
            data_prep=DataPrep(),
            mode=InputMode.BOTH
        ),
        transformer=PassThroughTransformer(),
        od_predictor=FCNN_OD(),
        parameters=GridsearchParams(),
        name="test_class_TEXT",
        res_folder="reports/one_class/"
    )
    grid_search(eval_run)
