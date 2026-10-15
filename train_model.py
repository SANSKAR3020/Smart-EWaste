import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
)
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_BASE = 10
EPOCHS_FINE = 15
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ProgressLog(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        flag = (
            " <-- overfitting, add more images"
            if train_acc - val_acc > 0.12
            else ""
        )
        print(
            f"Epoch {epoch + 1}: "
            f"train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"loss={logs.get('loss', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}{flag}",
            flush=True,
        )


def count_images(class_names):
    counts = {}

    for split in ("train", "val", "test"):
        counts[split] = {}

        for name in class_names:
            folder = DATA_DIR / split / name
            if not folder.is_dir():
                counts[split][name] = 0
                continue

            counts[split][name] = sum(
                1
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    return counts


def resolve_classes():
    train_dir = DATA_DIR / "train"

    if not train_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {train_dir}")

    all_folders = sorted(
        p.name for p in train_dir.iterdir() if p.is_dir()
    )
    counts = count_images(all_folders)

    used = [
        name for name in all_folders
        if counts["train"][name] > 0
    ]
    skipped = [
        name for name in all_folders
        if counts["train"][name] == 0
    ]

    if skipped:
        print(
            "Classes with no training images:",
            ", ".join(skipped),
        )

    return used, counts, skipped


def warn_small_classes(class_names, counts):
    problems = []

    for name in class_names:
        if counts["train"][name] < 50:
            problems.append(
                f"{name}: {counts['train'][name]} training images"
            )

        if counts["val"][name] < 10:
            problems.append(
                f"{name}: {counts['val'][name]} validation images"
            )

        if counts["test"][name] == 0:
            problems.append(f"{name}: no test images")

    if problems:
        print("\nDataset warnings:")
        for problem in problems:
            print(" -", problem)
        print("Accuracy may be unreliable until you add more images.\n")


def build_model(num_classes):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
        include_top=False,
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # MobileNetV2 expects image values scaled to [-1, 1].
    # This Keras layer avoids the KerasTensor preprocessing error.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(inputs)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="ewaste_mobilenetv2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base


def make_ds(split, class_names, training):
    split_dir = DATA_DIR / split

    for name in class_names:
        folder = split_dir / name
        if not folder.is_dir():
            raise SystemExit(
                f"Missing folder: {folder}\n"
                "Create this category folder before training."
            )

    counts = count_images(class_names)
    total = sum(counts[split].values())

    if total == 0:
        raise SystemExit(
            f"No images found in dataset/{split}/. "
            "Add images before running training."
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=SEED,
    )

    if training:
        augment = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.10),
                layers.RandomZoom(0.10),
                layers.RandomContrast(0.10),
            ],
            name="augment",
        )

        ds = ds.map(
            lambda images, labels: (
                augment(images, training=True),
                labels,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    return ds.prefetch(AUTOTUNE)


def evaluate(model, ds_test, class_names):
    y_true = []
    y_pred = []

    for images, labels in ds_test:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    if not y_true:
        print("No test images available.")
        return None

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    accuracy = float(np.mean(np.array(y_pred) == np.array(y_true)))

    print("\n--- Test Results ---")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print(f"Overall test accuracy: {accuracy:.4f}")

    text = (
        f"Class order: {class_names}\n\n"
        f"Confusion matrix:\n{matrix}\n\n"
        f"{report}\n"
        f"Overall test accuracy: {accuracy:.4f}\n"
    )

    (MODELS_DIR / "test_report.txt").write_text(
        text,
        encoding="utf-8",
    )

    return accuracy


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class_names, counts, skipped = resolve_classes()

    if len(class_names) < 2:
        raise SystemExit(
            "At least two categories must contain training images."
        )

    warn_small_classes(class_names, counts)

    print(f"\nTraining categories: {class_names}\n")
    print(f"{'Class':<12}{'Train':>8}{'Val':>8}{'Test':>8}")

    for name in class_names:
        print(
            f"{name:<12}"
            f"{counts['train'][name]:>8}"
            f"{counts['val'][name]:>8}"
            f"{counts['test'][name]:>8}"
        )

    ds_train = make_ds("train", class_names, True)
    ds_val = make_ds("val", class_names, False)

    model, base = build_model(len(class_names))

    print("\n--- Stage 1: Train classification head ---")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS_BASE,
        callbacks=[ProgressLog()],
    )

    print("\n--- Stage 2: Fine-tune MobileNetV2 ---")
    base.trainable = True

    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS_FINE,
        callbacks=[
            ProgressLog(),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            ),
        ],
    )

    model_path = MODELS_DIR / "ewaste_model.keras"
    model.save(model_path)

    (MODELS_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    (MODELS_DIR / "history.json").write_text(
        json.dumps(history.history),
        encoding="utf-8",
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved classes: {MODELS_DIR / 'class_names.json'}")

    ds_test = make_ds("test", class_names, False)
    evaluate(model, ds_test, class_names)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()