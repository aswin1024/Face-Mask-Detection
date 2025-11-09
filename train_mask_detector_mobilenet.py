# train_mask_detector_mobilenet.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (224, 224)
BATCH = 32
DATA_DIR = 'dataset'  # contains with_mask/ without_mask/

# Data generators with augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.6,1.4)
)

train_flow = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='training'
)

val_flow = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='validation'
)

# Base model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False  # freeze base for initial training

# Head
head = base.output
head = layers.GlobalAveragePooling2D()(head)
head = layers.Dropout(0.3)(head)
head = layers.Dense(128, activation='relu')(head)
head = layers.Dropout(0.3)(head)
preds = layers.Dense(1, activation='sigmoid')(head)

model = models.Model(inputs=base.input, outputs=preds)

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint("best_mask_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
]

# Train head
model.fit(train_flow, validation_data=val_flow, epochs=8, callbacks=callbacks)

# Fine-tune: unfreeze some layers
base.trainable = True
# unfreeze from a certain layer
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=8, callbacks=callbacks)

model.save('mask_detector_mobilenet_final.h5')
print("Saved mask_detector_mobilenet_final.h5")
