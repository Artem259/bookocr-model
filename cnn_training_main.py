import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import config as cg


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")), "\n")

class_names = [str(i) for i in range(len(cg.char_labels))]
train_data, validation_data = tf.keras.utils.image_dataset_from_directory(directory=cg.dataset_folder,
                                                                          labels="inferred",
                                                                          label_mode='int',
                                                                          class_names=class_names,
                                                                          color_mode="grayscale",
                                                                          batch_size=cg.batch_size,
                                                                          image_size=(cg.image_size, cg.image_size),
                                                                          shuffle=True,
                                                                          seed=cg.training_seed,
                                                                          validation_split=cg.validation_split,
                                                                          subset="both")
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_data = validation_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1/255)
train_data = train_data.map(lambda x, y: (rescale(x), y))
validation_data = validation_data.map(lambda x, y: (rescale(x), y))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(cg.char_labels), activation='softmax'))

model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

if os.path.isfile(cg.model_weights_file):
    model.load_weights(cg.model_weights_file)
    print("Loaded saved weights")
else:
    print("No saved weights found")


checkpoint = ModelCheckpoint(cg.model_weights_file, save_best_only=True, save_weights_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=cg.early_stopping_patience,
                           restore_best_weights=True, start_from_epoch=cg.early_stopping_start)
tensorboard = TensorBoard(log_dir=cg.logs_folder)
model.fit(train_data, epochs=cg.epochs_amount, validation_data=validation_data,
          callbacks=[checkpoint, early_stop, tensorboard])
print()
model.evaluate(validation_data, verbose=2)
model.save(cg.model_file)

print("\nCompleted.")
