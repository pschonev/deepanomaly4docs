from flair.data import Label
from scipy.sparse import data
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, TypeVar, List
from types import SimpleNamespace
import pandas as pd
import numpy as np
import keras
import sklearn
import pyod
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import concatenate
from src.utils import next_path, product_dict
from src.embedders import EmbeddingModel
import tensorflow as tf
import gc

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
NumpyArray = TypeVar('np.ndarray')


class RefType(Enum):
    SAME = "same"
    OTHER = "other"
    BOTH = "both"


class InputMode(Enum):
    TEXT = "text"
    IMAGE = "image"
    BOTH = "both" 


class EvalMode(Enum):
    ALL = "all"
    ONEOUT = "one_out"


class Input(BaseModel):
    name: str
    data_col: str
    vec_col: str

    embedder: EmbeddingModel

    target: NumpyArray = None
    ref: NumpyArray = None
    test: NumpyArray = None
    train: NumpyArray = None
    label: NumpyArray = None


class TextInput(Input):
    min_len: int = 250
    ref_data_path: str = ""
    ref_data: RefType = RefType.SAME


class ImageInput(Input):
    pass


class Labels(BaseModel):
    targets: str = "target"
    ref: NumpyArray = None
    train: NumpyArray = None
    test: NumpyArray = None

class DataPrep(BaseModel):
    """Loads, prepares, samples and holds the data from _one_ input file."""
    n_class: int = 20000
    contamination: float = 0.1
    fraction: float = 1.0
    seed: int = 42

    inliers: List[int] = [0, 1, 2, 11]
    outliers: List[int] = [4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
    unused_classes: List[int] = [3, 8]

    def remove_short_texts(self, df: PandasDataFrame, min_len: int) -> PandasDataFrame:
        n_before = df.shape[0]
        df = df[df['text'].map(len) > min_len]
        print(
            f"Removed {n_before - df.shape[0]} rows with doc length below {min_len}.")
        return df

    def sample_data(self, df: PandasDataFrame, target_col: str, mode:InputMode) -> PandasDataFrame:
        # remove unused classses
        df = df.where(~df[target_col].isin(self.unused_classes))
        df = df.dropna()
        # create inlier and outlier label column
        df["label"] = 0
        df.loc[df[target_col].isin(self.inliers), "label"] = 1
        
        if mode != InputMode.TEXT:
            print("Flatten image vectors")
            df.vecs = df.vecs.progress_map(lambda x: x.flatten())
        
        # get only n samples
        df = df.groupby(target_col, group_keys=False).apply(
            lambda df: df.sample(n=min(df.shape[0], self.n_class), random_state=42))
        df = df.reset_index(drop=True)
        
        # shuffle
        df = df.sample(frac=self.fraction)

        # apply contamination factor
        x_n = df[df.label == 1].shape[0]
        df = df[df["label"] == 1].head(x_n).append(
            df[df["label"] == 0].head(int(x_n*self.contamination)))
        
        return df.reset_index(drop=True)

    def split_data(self, df: PandasDataFrame, target_col: str):
        df, df_test = train_test_split(df, test_size=int(df.shape[0]*0.1), random_state=self.seed,
                                       stratify=df[target_col])
        return df, df_test

    def get_ref_data(self, df, txt: TextInput, mode: InputMode):
        # target data
        df_t = df.where(df.label == 1).dropna()

        # reference_data
        if mode == InputMode.TEXT:
            if txt.ref_data == RefType.OTHER:
                df_r = pd.read_pickle(self.ref_data.path)
                df_r = df_r.where(df_r.target != -1).dropna()
            elif txt.ref_data == RefType.BOTH:
                df_r = df.where(df.label == 0).dropna()
                df_other = pd.read_pickle(
                    txt.ref_data_path)
                df_other = df_other.where(df_other.target != -1).dropna()
                df_other.target = (df_other.target + 1)*20
                df_r = df_r.append(df_other)
            else:
                df_r = df.where(df.label == 0).dropna()
        else:
            if txt.ref_data == RefType.SAME or mode != InputMode.TEXT:
                df_r = df.where(df.label == 0).dropna()

        return df_r, df_t


class Database(BaseModel):
    path: str
    df: PandasDataFrame = None
    df_test: PandasDataFrame = None
    df_r: PandasDataFrame = None
    df_t: PandasDataFrame = None


    def __init__(self, **data: Any):
        super().__init__(**data)
        self.load_data()


    def load_data(self):
        print(f"Loading data from {self.path} to DataFrame...")
        self.df = pd.read_pickle(self.path)
        return self


    def vectorize(self, input_vec: Input) -> None:
        print(f"get train target vecs for {input_vec.name}")
        self.df_t[input_vec.vec_col] = input_vec.embedder.vectorize(
            self.df_t, data_col=input_vec.data_col)
        print(f"get train reference vecs {input_vec.name}")
        self.df_r[input_vec.vec_col] = input_vec.embedder.vectorize(
            self.df_r, data_col=input_vec.data_col)
        print(f"get test vecs {input_vec.name}")
        self.df_test[input_vec.vec_col] = input_vec.embedder.vectorize(
            self.df_test, data_col=input_vec.data_col)


    def vectorize_data(self, txt: TextInput, img: ImageInput, mode: InputMode) -> None:
        if mode == InputMode.TEXT or mode == InputMode.BOTH:
            self.vectorize(txt)
        if mode == InputMode.TEXT or mode == InputMode.BOTH:
            self.vectorize(img)


class DataHandler(BaseModel):
    database: Database
    txt_vecs: TextInput
    img_vecs: ImageInput
    labels: Labels

    data_prep: DataPrep

    mode: InputMode


    def prep_data(self) -> None:
        """Samples and vectorizes data and stores them in the DataBase object."""
        df = self.data_prep.remove_short_texts(self.database.df, self.txt_vecs.min_len)
        df = self.data_prep.sample_data(df, self.labels.targets, self.mode)
        print(f"after sample {df.columns}")
        df, self.database.df_test = self.data_prep.split_data(df, self.labels.targets)
        print(f"after split {df.columns}")
        self.database.df_r, self.database.df_t = self.data_prep.get_ref_data(df, self.txt_vecs, self.mode)
        print(f"df_t after get_ref_data {self.database.df_t.columns}")
    

    def prep_vecs(self, inliers: List[int], c:dict) -> None:
        """Selects data from database for given seed and settings and prepares the input vectors."""
        txt_vecs, img_vecs, labels = self.txt_vecs, self.img_vecs, self.labels
        df_t = self.database.df_t
        target = labels.targets

        if c.chosen_class is not None and txt_vecs.ref_data == RefType.SAME:
            if c.weakly:
                chosen_samples = self.database.df_r.where(self.database.df_r[target].isin(
                    c.chosen_class)).dropna().sample(n=c.weakly, random_state=c.random_state)

            df_r = self.database.df_r.where(~self.database.df_r[target].isin(c.chosen_class)).dropna()
            
            if c.weakly:
                df_r = df_r.append(chosen_samples)
        else:
            df_r = self.database.df_r

        print(f"df targets:\n{df_t[target].value_counts()}")
        print(f"df references (before remap):\n{df_r[target].value_counts()}")

        remap = {k: v for v, k in zip(
            range(df_r[target].unique().shape[0]), df_r[target].unique())}
        df_r.target = df_r[target].map(remap)

        print(f"df references (after remap):\n{df_r[target].value_counts()}")

        if c.chosen_class is not None:
            if c.balanced:
                df_test_out = self.database.df_test.where(
                    (self.database.df_test.label == 0) & (self.database.df_test[target].isin(c.chosen_class))).dropna()
                n = df_test_out.shape[0]
                df_test_in = self.database.df_test.where(self.database.df_test.label == 1)
                df_test_in = df_test_in.dropna(how="all").sample(
                    n=n, random_state=c.random_state)
                df_test = df_test_out.append(df_test_in).dropna(how="all")
            else:
                df_test_out = self.database.df_test.where(
                    self.database.df_test[target].isin(c.chosen_class)).dropna()
                df_test_in = self.database.df_test.where(self.database.df_test[target].isin(inliers)).dropna(
                ).sample(n=df_test_out.shape[0]*4, random_state=c.random_state)
                df_test = pd.concat([df_test_in, df_test_out])
        else:
            df_test = self.database.df_test
        print(f"df test:\n{df_test[target].value_counts()}")

        # vecs
        y_ref = np.array(df_r[target].to_list())
        labels.ref = keras.utils.to_categorical(y_ref)

        img_vecs.target = np.array(df_t[img_vecs.vec_col].to_list())
        img_vecs.ref = np.array(df_r[img_vecs.vec_col].to_list())

        txt_vecs.target = np.array(df_t[txt_vecs.vec_col].to_list())
        txt_vecs.ref = np.array(df_r[txt_vecs.vec_col].to_list())

        img_vecs.test = np.array(df_test[img_vecs.vec_col].to_list())
        txt_vecs.test = np.array(df_test[txt_vecs.vec_col].to_list())
        labels.test = np.array(df_test["label"].to_list())

        df_r_temp = df_r.groupby('target', group_keys=False).apply(
            lambda df: df.sample(n=min(df.shape[0], c.n_per_targ), random_state=42))

        img_vecs.train = np.array(df_t.head(c.n_sup).append(df_r_temp)[img_vecs.vec_col].to_list())
        txt_vecs.train = np.array(df_t.head(c.n_sup).append(
            df_r_temp)[txt_vecs.vec_col].to_list())
        labels.train = np.array(df_t.head(c.n_sup).append(df_r_temp).label.to_list())

    def get_train_test_vecs(self):
        if  self.mode == InputMode.BOTH:
            train, test = [self.img_vecs.train, self.txt_vecs.train], [self.img_vecs.test, self.txt_vecs.test]
        elif  self.mode == InputMode.TEXT:
            train, test = [self.txt_vecs.train], [self.txt_vecs.test]
        elif  self.mode == InputMode.IMAGE:
            train, test = [self.img_vecs.train], [self.img_vecs.test]

        y_tr = self.labels.train.astype(int)
        test_labels = self.labels.test.astype(int)

        return train, test, y_tr, test_labels




class EmbeddingTransformer(BaseModel, ABC):

    @abstractmethod
    def create_model(self, data:DataHandler, c:dict, dropout_rate:float=0.3) -> None:
        pass

    @abstractmethod
    def train_step(self, data: DataHandler, c:dict) -> None:
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

class FCNN(BaseModel):
    layers: List[int]

    def create_partial_model(self, data:DataHandler, create_out=False, dropout_rate:float=0.3):
        if data.mode == InputMode.TEXT or data.mode == InputMode.BOTH:
            text_input = keras.layers.Input(shape=(data.txt_vecs.target[0].shape[0],), name='text_input')
            text = keras.layers.Dropout(dropout_rate)(text_input)
            text_output = keras.layers.Dense(300, activation='relu')(text)
        if data.mode == InputMode.TEXT:
            x = text_output

        if data.mode == InputMode.IMAGE or data.mode == InputMode.BOTH:
            img_input = keras.layers.Input(shape=(data.img_vecs.target[0].shape[0],), name='img_input')
            img = keras.layers.Dropout(dropout_rate)(img_input)
            img_output = keras.layers.Dense(300, activation='relu')(img)
        if data.mode == InputMode.IMAGE:
            x = img_output

        if data.mode == InputMode.BOTH:
            x = keras.layers.concatenate([img_output, text_output], name='Concatenate')


        x = keras.layers.Dense(self.layers[0], activation='relu')(x)
        for neurons in self.layers[1:]:
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.Dense(neurons, activation='relu')(x)
        
        if create_out:
            x = keras.layers.Dense(1, activation='sigmoid')(x)

        if data.mode == InputMode.TEXT:
            inputs = text_input
        elif data.mode == InputMode.IMAGE:
            inputs = img_input
        elif data.mode == InputMode.BOTH:
            inputs = [img_input, text_input]
        

        model = keras.models.Model(inputs=inputs, outputs=x,
                            name='Final_output')                    
        model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


class PassThroughTransformer(EmbeddingTransformer):
    def create_model(self, data:DataHandler, c:dict, dropout_rate:float=0.3) -> None:
        pass

    def train_step(self, data: DataHandler, c:dict) -> None:
        pass

    def transform(self, X):
        return X

    def cleanup(self) -> None:
        pass


class OneClassTransformer(EmbeddingTransformer, FCNN):
    layers: List[int] = [256, 128, 64]
    model_t: Any = None
    model_r: Any = None

    loss: List[float] = Field(default_factory=list)
    loss_c: List[float] = Field(default_factory=list)
    epoch: int = 0 

    def create_loss(self, classes, batchsize):
        def original_loss(y_true, y_pred):
            lc = 1/(classes*batchsize) * batchsize**2 * keras.backend.sum((y_pred -
                                                            keras.backend.mean(y_pred, axis=0))**2, axis=[1]) / ((batchsize-1)**2)
            return lc
        return original_loss


    def create_model(self, data:DataHandler, c:dict, dropout_rate:float=0.3) -> None:
        shared_model = self.create_partial_model(data)

        if c.chosen_class is None:
            classes = data.database.df_r.target.unique().shape[0]
        else:
            classes = data.database.df_r.target.unique().shape[0] - len(c.chosen_class)


        model_t = keras.models.Model(inputs=shared_model.input, outputs=shared_model.output)

        model_r = keras.engine.network.Network(inputs=model_t.input,
                        outputs=model_t.output,
                        name="shared_layer")

        prediction = keras.layers.Dense(classes, activation='softmax')(model_t.output)
        model_r = keras.models.Model(inputs=model_r.input, outputs=prediction)

        prediction_t = keras.layers.Dense(c.feature_out, activation='softmax')(model_t.output)
        model_t = keras.models.Model(inputs=model_t.input, outputs=prediction_t)

        # get the loss for compactness
        original_loss = self.create_loss(classes, c.batchsize)
        optimizer = keras.optimizers.Adam(learning_rate=5e-5)

        model_t.compile(optimizer=optimizer, loss=original_loss)
        model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
        

        model_t.summary()
        model_r.summary()

        self.loss, self.loss_c, self.epoch = [], [], 0

        self.model_r, self.model_t = model_r, model_t


    def train_step(self, data: DataHandler, c:dict):
        x_r, x_t_r, y_r, lc, ld = [], [], [], [], []

        if data.mode == InputMode.IMAGE:
            ref_samples = np.arange(data.img_vecs.ref.shape[0])
        else:
            ref_samples = np.arange(data.txt_vecs.ref.shape[0])

        np.random.seed(self.epoch)

        if data.mode == InputMode.BOTH:
            np.random.shuffle(data.img_vecs.target)
            np.random.shuffle(data.txt_vecs.target)

            np.random.shuffle(ref_samples)
            for i in range(len(data.img_vecs.ref)):
                x_r.append(data.img_vecs.ref[ref_samples[i]])
                x_t_r.append(data.txt_vecs.ref[ref_samples[i]])
                y_r.append(data.labels.ref[ref_samples[i]])
            x_r = np.array(x_r)
            x_t_r = np.array(x_t_r)
            y_r = np.array(y_r)

            for i in range(int(len(data.img_vecs.target) / c.batchsize)):
                batch_target = data.img_vecs.target[i*c.batchsize:i*c.batchsize+c.batchsize]
                batch_text_target = data.txt_vecs.target[i *
                                                    c.batchsize:i*c.batchsize+c.batchsize]
                batch_ref = x_r[i*c.batchsize:i*c.batchsize+c.batchsize]
                batch_text_ref = x_t_r[i*c.batchsize:i*c.batchsize+c.batchsize]
                batch_y = y_r[i*c.batchsize:i*c.batchsize+c.batchsize]

                lc.append(self.model_t.train_on_batch([batch_target, batch_text_target],
                                                    np.zeros((c.batchsize, c.feature_out))))

                # reference data
                ld.append(self.model_r.train_on_batch(
                    [batch_ref, batch_text_ref], batch_y))

        if  data.mode == InputMode.TEXT:
            x_target, x_ref, y_ref = data.txt_vecs.target, data.txt_vecs.ref, data.labels.ref
        if  data.mode == InputMode.IMAGE:
            x_target, x_ref, y_ref = data.img_vecs.target, data.img_vecs.ref, data.labels.ref

        if data.mode != InputMode.BOTH:
            np.random.shuffle(x_target)

            np.random.shuffle(ref_samples)
            for i in range(len(x_ref)):
                x_r.append(x_ref[ref_samples[i]])
                y_r.append(y_ref[ref_samples[i]])
            x_r = np.array(x_r)
            y_r = np.array(y_r)

            for i in range(int(len(x_target) / c.batchsize)):
                batch_target = x_target[i*c.batchsize:i*c.batchsize+c.batchsize]
                batch_ref = x_r[i*c.batchsize:i*c.batchsize+c.batchsize]
                batch_y = y_r[i*c.batchsize:i*c.batchsize+c.batchsize]
                # target data
                lc.append(self.model_t.train_on_batch([batch_target],
                                                    np.zeros((c.batchsize, c.feature_out))))

                # reference data
                ld.append(self.model_r.train_on_batch(
                    [batch_ref], batch_y))
            
        self.epoch += 1    
        self.loss.append(np.mean(ld))
        self.loss_c.append(np.mean(lc))
        
        print(
            f"-----\n\nepoch : {self.epoch+1} ,Descriptive loss : {self.loss[-1]}, Compact loss : {self.loss_c[-1]}")
        return self


    def transform(self, X):
        return self.model_t.predict(X)

    def cleanup(self):
        """cleanup models""" 
        del self.model_r
        del self.model_t 
        gc.collect()
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    

class SupOutlierDetector(BaseModel, ABC):
    @abstractmethod
    def fit_predict(self, c:dict, train:list, test:list, y_tr, data:DataHandler, pass_through:bool):
        pass

def unpack_train_test(train, test, data):
    if data.mode == InputMode.BOTH:
        train, test = np.concatenate((train[0], train[1]), axis=1),  np.concatenate((test[0], test[1]), axis=1)
    else:
        train, test = train[0], test[0]
    return train, test


class SVM_OD(SupOutlierDetector):
    def fit_predict(self, c:dict, train:list, test:list, y_tr, data:DataHandler, pass_through:bool):
        train, test = unpack_train_test(train, test, data)
        print("fit SVM...")
        clf = sklearn.svm.SVC(probability=True)
        clf.fit(train, y_tr)

        decision_scores = clf.predict_proba(test)
        decision_scores = np.array([x[1] for x in decision_scores])
        return decision_scores


class OCSVM_OD(SupOutlierDetector):
    def fit_predict(self, c:dict, train:list, test:list, y_tr, data:DataHandler, pass_through:bool):
        train, test = unpack_train_test(train, test, data)
        print("fit OCSVM...")
        clf = pyod.models.ocsvm.OCSVM()
        clf.fit(train)
        decision_scores = clf.score_samples(test)
        return decision_scores


class FCNN_OD(SupOutlierDetector, FCNN):
    layers: List[int] = [256, 128, 64]

    def create_linear_probe_model(self, n_in, dropout_rate=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(n_in, input_dim=n_in, activation='relu'))
        for _ in self.layers:
            model.add(keras.layers.Dropout(dropout_rate))
            model.add(keras.layers.Dense(int(n_in/2), activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss="binary_crossentropy",
                    optimizer='adam', metrics=['accuracy'])
        return model

    
    def fit_predict(self, c, train, test, y_tr, data, pass_through):
        if pass_through:
            clf = self.create_partial_model(data, create_out=pass_through)
        else:
            clf = self.create_linear_probe_model(n_in=c.feature_out)
        clf.summary()
        clf.fit(
            train, y=y_tr, epochs=c.sup_epochs, batch_size=64, verbose=True)

        decision_scores = clf.predict(test)
        decision_scores = decision_scores.astype(float)

        # cleanup supervised model
        del clf
        gc.collect()
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        return decision_scores


class GridsearchParams(BaseModel):
    weakly: List[bool] = [None]
    batchsize: List[int] = [128]
    epoch_num: List[int] = [12]
    epoch_report: List[int] = [4]
    sup_epochs: List[int] = [15]
    feature_out: List[int] = [64]
    threshold: List[float] = [0.55]
    n_sup: List[int] = [10000] # samples per inlier class for final fcnn
    n_per_targ: List[int] = [1000] # samples per outlier class (reference data) for fcnn
    random_state: List[int] = list(range(1,5))
    balanced: List[bool] = [False]
    use_umap: List[bool] = [False]
    chosen_class: List[List[int]] = None

    def param_combinations(self, eval_mode, outliers, transformer):
        if isinstance(transformer, PassThroughTransformer):
            self.epoch_num = [1]

        if eval_mode == EvalMode.ALL:
            self.chosen_class = [None]
        elif eval_mode == EvalMode.ONEOUT:
            self.chosen_class = [[x] for x in outliers]

        return [SimpleNamespace(**x) for x in product_dict(**self.__dict__)]


class SupEvalRun(BaseModel):
    eval_mode: EvalMode

    data: DataHandler
    transformer: EmbeddingTransformer
    od_predictor: SupOutlierDetector
    parameters: GridsearchParams

    name: str
    res_folder: str = ""
    res_path: str = ""


    def init_result_path(self) -> None:
        self.res_path = next_path(
            self.res_folder + "%04d_" + self.name + ".tsv")
        print(f"Saving results to {self.res_path}")
        return self
