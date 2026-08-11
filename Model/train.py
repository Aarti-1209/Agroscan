import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15            # phase 1 (frozen base) — increased from 10
FINE_TUNE_EPOCHS = 8   # phase 2 (fine-tuning last layers)
DATASET_PATH = "dataset/plantvillage dataset/color"

# Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

NUM_CLASSES = len(train_data.class_indices)
print(f"Total Classes: {NUM_CLASSES}")

# MobileNetV2 model
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

# Custom layers
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── Phase 1: Train with frozen base ───────────────────────────
print("\n=== Phase 1: Training with frozen MobileNetV2 base ===\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'model/plant_disease.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
)

best_phase1_acc = max(history.history['val_accuracy'])
print(f"\nPhase 1 complete. Best validation accuracy: {best_phase1_acc:.2%}")

# ─── Phase 2: Fine-tune last layers of the base model ──────────
print("\n=== Phase 2: Fine-tuning last 30 layers of MobileNetV2 ===\n")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # much lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'model/plant_disease_finetuned.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
)

best_phase2_acc = max(history_finetune.history['val_accuracy'])
print(f"\nPhase 2 complete. Best validation accuracy: {best_phase2_acc:.2%}")

# ─── Keep whichever phase actually performed better ────────────
if best_phase2_acc > best_phase1_acc:
    model.save('model/plant_disease.h5')  # overwrite with the fine-tuned (better) model
    print(f"\n✅ Fine-tuning IMPROVED the model: {best_phase1_acc:.2%} -> {best_phase2_acc:.2%}")
    print("Final model saved: model/plant_disease.h5 (fine-tuned version)")
else:
    print(f"\n⚠️ Fine-tuning did NOT improve on phase 1 ({best_phase1_acc:.2%} vs {best_phase2_acc:.2%})")
    print("Keeping phase 1 model as final: model/plant_disease.h5 (already saved by checkpoint)")

# Class names save karo
with open('model/class_names.json', 'w') as f:
    json.dump(train_data.class_indices, f)

final_acc = max(best_phase1_acc, best_phase2_acc)
print(f"\nTraining Complete!")
print(f"Best Overall Accuracy: {final_acc:.2%}")