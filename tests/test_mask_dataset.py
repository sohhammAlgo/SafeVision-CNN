import tensorflow as tf

from src.data.mask_dataset import (
    load_mask_datasets,
    preprocess_dataset,
    get_class_weights
)


def test_class_weights():

    weights = get_class_weights()

    assert len(weights) == 3

    assert weights[2] > weights[1]

    assert weights[1] > weights[0]


def test_dataset_pipeline():

    train_ds, val_ds, test_ds = (
        load_mask_datasets()
    )

    train_ds, val_ds, test_ds = (
        preprocess_dataset(
            train_ds,
            val_ds,
            test_ds
        )
    )

    images, labels = next(
        iter(train_ds)
    )

    assert images.shape[1:] == (
        224,
        224,
        3
    )

    assert labels.shape[0] <= 32