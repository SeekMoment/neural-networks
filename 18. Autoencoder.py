import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

input_img = Input(shape=(28, 28, 1))
x = Flatten()(input_img)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
encoded = Dense(2, activation='relu')(x)

input_enc = Input(shape=(2,))
d = Dense(64, activation='relu')(input_enc)
d = Dense(28 * 28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d)

encoder = keras.Model(input_img, encoded, name='encoder')
decoder = keras.Model(input_enc, decoded, name='decoder')
autoencoder = keras.Model(input_img, decoder(encoder(input_img)), name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train,
                 epochs=10, batch_size=64, shuffle=True)

h = encoder.predict(x_test)

fig = plt.figure(figsize=(7,4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(h[:, 0], h[:, 1])

img = decoder.predict(np.expand_dims([50, 250], axis=0))
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img.squeeze(), cmap='gray')
plt.show()



