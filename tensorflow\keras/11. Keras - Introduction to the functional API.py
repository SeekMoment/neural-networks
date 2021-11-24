import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Conv2DTranspose
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
import matplotlib.pyplot as plt

tf.random.set_seed(1)


# input = Input(shape=(32, 32, 3))
# x = Conv2D(32, 3, activation='relu')(input)
# x = MaxPooling2D(2, padding='same')(x)
# x = Conv2D(64, 3, activation='relu')(x)
# x = MaxPooling2D(2, padding='same')(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(10, activation='softmax')(x)
#
# model = keras.Model(inputs=input, outputs=output)
# model.summary()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
# model.compile(optimizer='adam', loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
#
# print(model.evaluate(x_test, y_test))

# class TfConv2D(tf.Module):
#     def __init__(self, kernel=(3, 3), channels=1, strides=(2, 2), padding='SAME', activate='relu'):
#         super(TfConv2D, self).__init__()
#         self.kernel = kernel
#         self.channels = channels
#         self.strides = strides
#         self.padding = padding
#         self.activate = activate
#         self.fl_init = False
#
#     def __call__(self, x):
#         if not self.fl_init:
#             # [kernel_x, kernel_y, input_channels, output_channels]
#             self.w = tf.random.truncated_normal((*self.kernel, x.shape[-1], self.channels), stddev=0.1, dtype=tf.double)
#             self.b = tf.zeros([self.channels], dtype=tf.double)
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b)
#
#             self.fl_init = True
#
#         y = tf.nn.conv2d(x, self.w, strides=(1, *self.strides, 1), padding=self.padding) + self.b
#
#         if self.activate == 'relu':
#             return tf.nn.relu(y)
#         elif self.activate == 'softmax':
#             return tf.nn.softmax(y)
#
#         return y
#
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
# layer1 = TfConv2D((3, 3), 32)
# print(x_test[0])
# y = layer1(tf.expand_dims(x_test[0], axis=0))
# y = tf.nn.max_pool2d(y, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
# print(y.shape)



enc_input = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation='relu')(enc_input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Flatten()(x)
enc_output = Dense(8, activation='linear')(x)

encoder = keras.Model(inputs=enc_input, outputs=enc_output, name='encoder')

dec_input = Input(shape=(8,), name='encoded_img')
x = Dense(7*7*8, activation='relu')(dec_input)
x = Reshape((7, 7, 8))(x)
x = Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(32, 5, strides=(2, 2), activation='linear', padding='same')(x)
x = BatchNormalization()(x)
dec_output = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='decoder')


autoencoder_input = Input(shape=(28, 28, 1), name='img')
x = encoder(autoencoder_input)
autoencoder_output = decoder(x)

autoencoder = keras.Model(autoencoder_input, autoencoder_output, name='autoencoder')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, batch_size=32, epochs=1)


h = encoder.predict(tf.expand_dims(x_test[0], axis=0))
img = decoder.predict(h)

plt.subplot(121)
plt.imshow(x_test[0], cmap='gray')
plt.subplot(122)
plt.imshow(img.squeeze(), cmap='gray')
plt.show()






