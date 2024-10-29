import pytest
import sys
import logging
import pathlib

from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import DeepPurpose.dataset
import DeepPurpose.utils

sys.path.insert(0, ".")  # TODO: why is it needed?
import data_process


@pytest.fixture
def davis_data(tmpdir):
    dir_data = tmpdir/'data'

    X_drug, X_target, affinity = DeepPurpose.dataset.load_process_DAVIS(
        str(dir_data),
        binary=False,
    )

    return X_drug, X_target, affinity


@pytest.fixture
def bipartite_davis_data(tmpdir, davis_data):
    dir_data = tmpdir/'data'

    X_target = pd.read_json(dir_data/'DAVIS/target_seq.txt', typ='series')
    X_drug = pd.read_json(dir_data/'DAVIS/SMILES.txt', typ='series')

    affinity = pd.read_csv(
        dir_data/'DAVIS/affinity.txt',
        sep = ' ',
        names=X_target.index,
    )
    affinity.index = X_drug.index

    return X_drug, X_target, affinity


def test_kiba_pandas_preprocessing_for_deeppurpose():
    X_drug = pd.read_table("../datasets/kiba/final/smiles.tsv", index_col=0)
    X_target = pd.read_table("../datasets/kiba/final/target_sequences.tsv", index_col=0)
    y = pd.read_table("../datasets/kiba/final/affinity.tsv", index_col=0)

    drug_encoding = 'CNN'
    target_encoding = 'CNN'

    gso_data = data_process.preprocess_dti_for_deeppurpose(
        [X_drug, X_target],
        y,
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
    )


def test_kiba_preprocessing_for_deeppurpose():
    sys.path.insert(0, "..")
    import data_loading

    X_drug = data_loading.read_table_to_array("../datasets/kiba/final/smiles.tsv")
    X_target = data_loading.read_table_to_array("../datasets/kiba/final/target_sequences.tsv")
    y = data_loading.load_log_affinities("../datasets/kiba/final/affinity.tsv")

    drug_encoding = 'CNN'
    target_encoding = 'CNN'

    gso_data = data_process.preprocess_dti_for_deeppurpose(
        [X_drug, X_target],
        y,
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
    )


def test_davis_preprocessing_for_deeppurpose(davis_data, bipartite_davis_data):
    X_drug, X_target, affinity = davis_data
    X_drug2, X_target2, affinity2 = bipartite_davis_data

    drug_encoding = 'CNN'
    target_encoding = 'CNN'

    gso_data = DeepPurpose.utils.data_process(
        X_drug,
        X_target,
        affinity,
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        split_method='no_split',
    )

    gso_data2 = data_process.preprocess_dti_for_deeppurpose(
        [X_drug2.values, X_target2.values],
        affinity2.values,
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        convert_to_log=True,
    )

    assert (gso_data == gso_data2).all().all()


@pytest.mark.parametrize("under_sampler", [None, RandomUnderSampler()])
def test_deep_purpose_wrapper(bipartite_davis_data, under_sampler):
    from .deep_purpose_wrapper import DeepPurposeWrapper

    config = DeepPurpose.utils.generate_config(
        drug_encoding = 'CNN',
        target_encoding = 'CNN',
        cls_hidden_dims = [1024,1024,512],
        train_epoch = 10,
        LR = 0.001,
        batch_size = 256,
        cnn_drug_filters = [32,64,96],
        cnn_target_filters = [32,64,96],
        cnn_drug_kernels = [4,6,8],
        cnn_target_kernels = [4,8,12],
    )

    subsample = 8
    X_drug, X_target, affinity = bipartite_davis_data
    X_drug, X_target, affinity = (
        X_drug.iloc[:subsample],
        X_target.iloc[:subsample],
        affinity.iloc[:subsample, :subsample],
    )

    Xs = [X_drug.iloc[:11].values, X_target.iloc[:13].values]
    ys = affinity.iloc[:11, :13].values

    deep_dta = DeepPurposeWrapper(
        config,
        under_sampler=under_sampler,
        binarizer=lambda y: (y < y.max()).astype(int),
    )
    deep_dta.fit([X_drug, X_target], affinity)
    pred = deep_dta.predict([X_drug, X_target])
    assert pred.shape == (pred.size,)
    assert pred.size == affinity.size
    score = deep_dta.score(Xs, ys)
    logging.info(f"Score: {score}")

