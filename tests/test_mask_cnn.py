import tensorflow as tf

from src.models.mask_cnn import build_mask_cnn


def test_mask_cnn_output_shape():

    model = build_mask_cnn()

    assert model.output_shape == (
        None,
        3
    )