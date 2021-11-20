import tensorflow as tf
a = tf.zeros((3, 3))
print(a)

b = tf.ones((5, 3))
print(b)

c = tf.ones_like(a)
d = tf.zeros_like(a)

d = tf.eye(3)
print(d)

e = tf.identity(c)  # создает копию тензора

f = tf.fill((2, 3), -1)


g = tf.range(1, 11, 0.2)


a = tf.random.normal((2, 4), 0, 0.1)  # 0 - мат ожидание, 0.1 - дисперсия
print(a)

b = tf.random.uniform((2, 2), -1, 1)
print(b)


tf.random.set_seed(1)
d = tf.random.truncated_normal((1, 5), -1, 0.1)

a = tf.constant([1, 2, 3])
b = tf.constant([9, 8, 7])

print(tf.add(a, b))
# or
print(a + b)


print(tf.subtract(a, b))
print(a - b)


print(tf.divide(a, b))
print(a / b)

print(a // b)

print(tf.multiply(a, b))
print(a * b)

print(a ** 2)


print(tf.tensordot(a, b, axes=0))  # внешнее векторное произведение
print(tf.tensordot(a, b, axes=1))  # внутренее векторное произведение (скалярное)

a2 = tf.constant(tf.range(1, 10), shape=(3, 3))
b2 = tf.constant(tf.range(5, 14), shape=(3, 3))
print(tf.matmul(a2, b2))
# or
print(a2 @ b2)

print(tf.reduce_sum(a2))
# print(tf.reduce_mean(a2))
# print(tf.reduce_max(a2))
# print(tf.reduce_prod(a2))
# print(tf.sqrt(tf.cast(a2, dtype=tf.float32)))





