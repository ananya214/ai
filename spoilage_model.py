import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

train_dir = "Dataset/train"
test_dir = "Dataset/test"

IMG_SIZE = (150,150)
BATCH_SIZE = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(150,150,3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128,activation="relu")(x)
x = Dropout(0.5)(x)

predictions = Dense(1,activation="sigmoid")(x)

model = Model(inputs=base_model.input,outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_data,
    epochs=30,
    validation_data=test_data,
    callbacks=[early]
)

model.save("food_spoilage_model.h5")

print("Spoilage model trained!")