from src.data.mask_dataset import (
    load_mask_datasets,
    preprocess_dataset,
    get_class_weights
)


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


print("Image batch shape:")
print(images.shape)

print("\nLabel batch shape:")
print(labels.shape)

print("\nPixel minimum:")
print(float(images.numpy().min()))

print("\nPixel maximum:")
print(float(images.numpy().max()))

print("\nClass weights:")
print(get_class_weights())