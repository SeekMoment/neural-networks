import tensorflow as tf

x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2

df = tape.gradient(y, x)
print(df)

w = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
x = tf.Variable([[-2.0, 1.0, 3.0]], trainable=False)  # превращаем в тензор константу и по нему нельзя брать производные

#
# with tf.GradientTape(watch_accessed_variables=False) as tape:  # запретил считать по всем переменным производную
#     tape.watch(x)  # тут разрешаю конкретно для х
#     y = x @ w + b
#     loss = tf.reduce_mean(y ** 2)
#
#
# df = tape.gradient(y, [x, w])  # производная у по х and w
# print(df)
#
# with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:  # не осовобождаем ресурсы после градиента
#     tape.watch(x)  # тут разрешаю конкретно для х
#     y = x @ w + b
#     loss = tf.reduce_mean(y ** 2)
#
#
# df = tape.gradient(y, x)
# df_dw = tape.gradient(y, w)
#
# del tape
# print(df)


# x = tf.Variable([1.0, 2.0])
# with tf.GradientTape() as tape:
#     y = tf.reduce_sum([2.0, 3.0] * x ** 2)
# df = tape.gradient(y, x)
# print(df)



x = tf.Variable(1.0)

with tf.GradientTape() as tape:
    if x < 2.0:
        y = tf.reduce_sum([2.0, 3.0] * x ** 2)
    else:
        y = x ** 2

df = tape.gradient(y, x)
print(df)


