import numpy as np
import pytest
from hydra import compose, initialize

from src.dataset.base_dataset import BaseDataset
from src.dataset.load_dataset import get_dataset
from src.dataset.sample import Sample


@pytest.fixture
def dummy_sample():
    return Sample(
        sample_id=0,
        embeddings=np.ones((3, 4), dtype=np.float32),
        noisy_embeddings=np.ones((3, 4), dtype=np.float32) * 2,
        input_token_ids=np.array([1, 2, 3]),
        metadata={"input_text": "dummy", "_batch_idx": 0},
    )


@pytest.fixture
def config():
    with initialize(
        version_base=None, config_path="../config"
    ):  # path to your config directory
        cfg = compose(config_name="main.yaml")
    return cfg


@pytest.fixture
def dataset(config):
    return get_dataset(config)


def test_perturb_embeddings(dataset):
    sample = dataset[0]
    sample.perturb_embeddings(0.1)
    assert sample.noisy_embeddings is not None
    assert sample.noisy_embeddings.shape == sample.embeddings.shape


def test_dataset_creation(dataset):
    assert dataset is not None


def test_dataset_length(dataset):
    assert len(dataset) == 100


def test_dataset_getitem(dataset):
    sample = dataset[0]
    assert isinstance(sample, Sample)
    assert len(sample.embeddings.shape) == 2


def test_add_const_gaussian_noise(dataset: BaseDataset, config):
    config.privacy.noise_type = "constant"
    config.privacy.const_noise_params = {
        "true_mean": 0.0,
        "true_variance": 1.0,
        "noise_dist": "gaussian",
        "seed": 42,
        "load_from_file": False,
    }
    before = dataset[0].noisy_embeddings.copy()
    dataset.add_const_gaussian_noise(mean=0.0, variance=1.0)
    after = dataset[0].noisy_embeddings
    assert not np.allclose(before, after)
