import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# x = tf.Variable([[2.0]])
# y = tf.Variable([[-4.0]])
#
# with tf.GradientTape() as tape:
#     f = (x + y) ** 2 + 2 * x * y
#
# df = tape.gradient(f, [x, y])
# print(df[0], df[1])


x = tf.Variable(-1.0)
y = lambda: x ** 2 - x

N = 100
opt = tf.optimizers.SGD(learning_rate=0.1)
for n in range(N):
    opt.minimize(y, [x])

print(x.numpy())




