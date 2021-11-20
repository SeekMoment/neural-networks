import tensorflow as tf
import numpy as np

a = tf.constant(1, shape=(1, 1))
print(a)

b = tf.constant([1, 2, 3, 4])
print(b)

c = tf.constant([[1, 2, ],
                 [3, 4],
                 [5, 6]], dtype=tf.float32)
print(c)

a2 = tf.cast(a, dtype=tf.float32)
print(a2)

b1 = np.array(b)
# or
b2 = b.numpy()

v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
v3 = tf.Variable(b)

v1.assign(0)
v2.assign([0, 1, 6, 7])

v3.assign_add([1, 1, 1, 1])
v1.assign_sub(5)
print(v1, v2, v3)

print(v3.shape)

v4 = tf.Variable(v3)
val_0 = v4[0]
val_12 = v4[1:3]
val_0.assign(7)
print(val_0)
print(val_12)
print(v4)



x = tf.constant(range(10)) + 5
print(x)

x_indx = tf.gather(x, [0, 5])  # 0, 5 - это индексы
print(x)
print(x_indx)  # [5, 10]


print(c[0, 1])


a = tf.constant(range(30))
print(a)
b = tf.reshape(a, [5, 6])
b_T = tf.transpose(b, perm=[1, 0])
print(b_T)
print(b)
