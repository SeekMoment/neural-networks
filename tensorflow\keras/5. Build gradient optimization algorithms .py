import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)


k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)
batch_size = 100
num_steps = TOTAL_POINTS // batch_size

epochs = 50
learning_rate = 0.02

# optimizer = tf.optimizers.SGD(momentum=0.5, nesterov=True, learning_rate=0.02)
# optimizer = tf.optimizers.Adagrad(learning_rate=0.2)
# optimizer = tf.optimizers.Adadelta(learning_rate=4.0)
# optimizer = tf.optimizers.RMSprop(learning_rate=0.01)
optimizer = tf.optimizers.Adam(learning_rate=0.1)
for n in range(epochs):
    for n_batch in range(num_steps):
        y_batch = y[n_batch*batch_size:(n_batch+1) * batch_size]
        x_batch = x[n_batch*batch_size:(n_batch+1) * batch_size]

        with tf.GradientTape() as tape:
            f = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - f))
        dk, db = tape.gradient(loss, [k, b])

        optimizer.apply_gradients(zip([dk, db], [k, b]))
        # k.assign_sub(learning_rate * dk)
        # b.assign_sub(learning_rate * db)

print(k)
print(b)
y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()