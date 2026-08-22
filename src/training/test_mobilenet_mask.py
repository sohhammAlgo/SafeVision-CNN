from src.models.mobilenet_mask import build_mobilenet_mask


def test_mobilenet_output_shape():

    model = build_mobilenet_mask()

    assert model.output_shape == (
        None,
        3
    )


def test_mobilenet_backbone_is_frozen():

    model = build_mobilenet_mask()

    trainable_layers = [
        layer
        for layer in model.layers
        if layer.trainable
    ]

    assert len(trainable_layers) > 0