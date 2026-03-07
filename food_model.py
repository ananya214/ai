import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -------------------
# Dataset paths
# -------------------

train_dir = "food_dataset/train"
test_dir = "food_dataset/test"

IMG_SIZE = (150,150)
BATCH_SIZE = 16

# -------------------
# Strong Augmentation
# -------------------

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7,1.3],
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_data.num_classes

# -------------------
# Class Weights (Fix roti imbalance)
# -------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

# -------------------
# Base Model
# -------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(150,150,3)
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# -------------------
# Custom Head
# -------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)

predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------
# Compile
# -------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Callbacks
# -------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3
)

# -------------------
# Train
# -------------------

model.fit(
    train_data,
    epochs=40,
    validation_data=test_data,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# -------------------
# Save model
# -------------------

model.save("food_detection_model.keras")

print("Food detection model trained successfully!")