import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# model = keras.Sequential([
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# layer1 = Dense(128, activation='relu')
# layer2 = Dense(10, activation='softmax')
#
# x = tf.random.uniform((1, 20), 0, 1)
# y = layer2(layer1(x))
# print(y)

# model.pop()
# print(model.layers)
# model.add(Dense(5, activation='linear'))
# print(model.layers)


# x = tf.random.uniform((1, 20), 0, 1)
# y = model(x)
# print(model.weights)
# model.summary()


model = keras.Sequential([
    # Input(shape=(20,)),
    Dense(128, activation='relu', input_shape=(784,), name='hidden_1'),
    Dense(10, activation='softmax', name='output')
])

# model_ex = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
# model_ex = keras.Model(inputs=model.inputs, outputs=[model.layers[-1].output])
# model_ex = keras.Model(inputs=model.inputs, outputs=model.get_layer(name='output').output)


# print(model.weights)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)

# x = tf.expand_dims(x_train[0], axis=0)
# y = model_ex(x)
# y2 = model(x)
# print(y, y2)



model_ex = keras.Sequential([
    model,
    Dense(10, activation='tanh')
])

model.trainable = False

model_ex.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
model_ex.fit(x_train, y_train, batch_size=32, epochs=3)
