import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


# size_val = 10000
# x_val_split = x_train[:size_val]
# y_val_split = y_train_cat[:size_val]
#
# x_train_split = x_train[size_val:]
# y_train_split = y_train_cat[size_val:]

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)


plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
print(model.summary())


myAdam = keras.optimizers.Adam(learning_rate=0.1)
myOpt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

model.compile(optimizer=myOpt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_split, y_train_split, batch_size=32, epochs=10, validation_data=(x_val_split, y_val_split))


model.evaluate(x_test, y_test_cat)

n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f'Распознанная цифра: {np.argmax(res)}')

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
y_false = x_test[~mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()



model.save_weights('FFNN-MNIST.h5')