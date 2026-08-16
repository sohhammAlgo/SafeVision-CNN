import tensorflow as tf


def test_tensorflow_available():
    assert tf.__version__ is not None