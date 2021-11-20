import numpy as np

import matplotlib.pyplot as plt

N = 5
x1 = np.random.random(N)
x2 = x1 + [np.random.randint(10)/10 for i in range(N)]
c1 = [x1, x2]

x1 = np.random.random(N)
x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1
c2 = [x1, x2]

f = [0, 1]
w = np.array([-0.3, 0.3])
for i in range(N):
    x = np.array([c2[0][i], c2[1][i]])
    y = np.dot(w, x)
    if y >= 0:
        print('Класс С1')
    else:
        print('Класс С2')
plt.scatter(c1[0][:], c1[1][:], s=10, c='red')
plt.scatter(c2[0][:], c2[1][:], s=10, c='blue')
plt.plot(f)
plt.grid(True)
plt.show()



# N = 5
# b = 3
#
# x1 = np.random.random(N)
# x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b
# C1 = [x1, x2]
#
# x1 = np.random.random(N)
# x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 + b
# C2 = [x1, x2]
#
# f = [0+b, 1+b]
#
# w2 = 0.5
# w3 = -b*w2
# w = np.array([-w2, w2, w3])
# for i in range(N):
#     x = np.array([C1[0][i], C1[1][i], 1])
#     y = np.dot(w, x)
#     if y >= 0:
#         print("Класс C1")
#     else:
#         print("Класс C2")
#
# plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
# plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
# plt.plot(f)
# plt.grid(True)
# plt.show()




def act(x):
    return 0 if x <= 0 else 1

def go(C):
    x = np.array([C[0], C[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    sum = np.dot(w_hidden, x)
    out = [act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = act(sum)
    return y

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

print( go(C1[0]), go(C1[1]) )
print( go(C2[0]), go(C2[1]) )

