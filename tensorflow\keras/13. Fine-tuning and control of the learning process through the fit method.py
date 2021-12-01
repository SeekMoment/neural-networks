import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

tf.random.set_seed(1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28) / 255
x_test = x_test.reshape(-1, 28 * 28) / 255

sample_weight = np.ones(shape=(len(x_train)))
sample_weight[y_train == 1] = 5.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=32, epochs=5)
# model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

validation_split = 0.2
validation_split_index = np.ceil(x_train.shape[0] * validation_split).astype('int32')
x_train_val = x_train[:validation_split_index]
y_train_val = y_train[:validation_split_index]

x_train_data = x_train[validation_split_index:]
y_train_data = y_train[validation_split_index:]

# model.fit(x_train_data, y_train_data, batch_size=32, epochs=5, validation_data=(x_train_val, y_train_val))


train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data, y_train_data))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_train_val, y_train_val))
val_dataset = val_dataset.batch(64)


# model.fit(train_dataset, epochs=5, validation_data=val_dataset)
# model.fit(train_dataset, epochs=5, steps_per_epoch=100, validation_data=val_dataset)
# model.fit(train_dataset, epochs=5, validation_data=val_dataset, validation_steps=5)


# class_weight = {
#     0: 1.0,
#     1: 1.0,
#     2: 1.0,
#     3: 1.0,
#     4: 1.0,
#     5: 1.0,
#     6: 1.0,
#     7: 1.0,
#     8: 1.0,
#     9: 1.0,
# }
# model.fit(train_dataset, epochs=5, validation_data=val_dataset, class_weight=class_weight)


# model.fit(x_train, y_train, epochs=5, sample_weight=sample_weight)


# history = model.fit(x_train, y_train, epochs=3, validation_split=0.3)
# print(history.history)


# class DigitsLimit(keras.utils.Sequence):
#     def __init__(self, x, y, batch_size, max_len=-1):
#         self.batch_size = batch_size
#         self.x = x[:max_len]
#         self.y = y[:max_len]
#
#     def __len__(self):
#         return int(np.ceil(self.x.shape[0] / self.batch_size))
#
#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#
#         return batch_x, batch_y
#
#     def on_epoch_end(self):
#         p = np.random.permutation(len(self.x))
#         self.x = self.x[p]
#         self.y = self.y[p]
#         print("on_epoch_end")
#
#
# sequence = DigitsLimit(x_train, y_train, 64)
# history = model.fit(sequence, epochs=3)


# callbacks = [keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, verbose=1),
#              keras.callbacks.ModelCheckpoint(filepath="mymodel_{epoch}", save_best_only=True, monitor="loss", verbose=1),]
# model.fit(x_train, y_train, callbacks=callbacks)
#
#
# model = keras.models.load_model('mymodel_1')
# print(model.evaluate(x_test, y_test))


class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_train_end(self, logs):
        print(self.per_batch_losses[:5])


callbacks = [
    CustomCallback(),
]
model.fit(x_train, y_train, epochs=3, callbacks=callbacks)
