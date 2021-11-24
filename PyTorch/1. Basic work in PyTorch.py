import torch

print(torch.zeros([3, 4]))

print(torch.ones([3, 4, 3]))

print(torch.tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]]))

x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# print(x.size())
#
# print(x.shape)
#
#
# print(x[0])
# print(x[1])
# print(x[0, 0])
# print(x[:, 0])
#
# print(x + 10)
#
# print(x ** 2)
#
# y = torch.tensor([[12, 11, 10, 9],
#                   [8, 7, 6, 5],
#                   [4, 3, 2, 1]])
#
# print(x + y)
# print(x / y)
# print(x % y)

# print(torch.exp(x))
# print(torch.log(x))
# print(torch.sin(x))

print(x > 3)
mask = x > 3
print(x[mask])
print(x[x > 3])



# task

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())
larger_than_limit_sum = X[X > limit].sum()
print(larger_than_limit_sum)


# Копирование
# y = x
# y[0, 0] = 999
# print(x)
# print(y)

y = x.clone()
y[0, 0] = 999
print(x)
print(y)

print(x.dtype)


x = x.double()
print(x)
x = x.int()
print(x)
x = x.float()
print(x)


import numpy as np
x = np.array([[1, 2, 3, 4],
              [4, 3, 2, 1]])
print(x)

x = torch.from_numpy(x)
print(x)

x = x.numpy()
print(x)


x = torch.rand([2, 3])

torch.cuda.is_available()

torch.device('cuda:0')
torch.device('cpu')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x_cuda = x.to(device)
print(x_cuda)

# С большим количеством данных cuda быстрее
