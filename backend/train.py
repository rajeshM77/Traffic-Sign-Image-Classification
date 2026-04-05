import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
IMG_SIZE    = 32
NUM_CLASSES = 43
BATCH_SIZE  = 64
EPOCHS      = 30
DATA_DIR    = "data/gtsrb/Train"
MODEL_PATH  = "models/traffic_sign_model.h5"

# ── Load images ─────────────────────────────────────────
print("Loading images...")
images, labels = [], []

for class_id in range(NUM_CLASSES):
    class_folder = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_folder):
        print(f"  Warning: folder {class_folder} not found, skipping")
        continue
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(class_id)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")

images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)
print(f"Loaded {len(images)} images across {NUM_CLASSES} classes")

# ── Train/validation split ───────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES)

# ── Data augmentation ────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# ── Model architecture ───────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_model()
model.summary()

# ── Compile ──────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
]

# ── Train ────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── Save class names ─────────────────────────────────────
class_names = {
    0:"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)",
    3:"Speed limit (60km/h)", 4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)",
    6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)", 8:"Speed limit (120km/h)",
    9:"No passing", 10:"No passing veh over 3.5 tons", 11:"Right-of-way at intersection",
    12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles",
    16:"Veh > 3.5 tons prohibited", 17:"No entry", 18:"General caution",
    19:"Dangerous curve left", 20:"Dangerous curve right", 21:"Double curve",
    22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on the right",
    25:"Road work", 26:"Traffic signals", 27:"Pedestrians", 28:"Children crossing",
    29:"Bicycles crossing", 30:"Beware of ice/snow", 31:"Wild animals crossing",
    32:"End speed + passing limits", 33:"Turn right ahead", 34:"Turn left ahead",
    35:"Ahead only", 36:"Go straight or right", 37:"Go straight or left",
    38:"Keep right", 39:"Keep left", 40:"Roundabout mandatory",
    41:"End of no passing", 42:"End no passing veh > 3.5 tons"
}
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

# ── Plot training history ────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],     label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.title("Accuracy"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],     label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Loss"); plt.legend()

plt.tight_layout()
plt.savefig("models/training_history.png")
print("\nTraining history saved to models/training_history.png")
print(f"Model saved to {MODEL_PATH}")
print(f"\nFinal val accuracy: {max(history.history['val_accuracy']):.4f}")
