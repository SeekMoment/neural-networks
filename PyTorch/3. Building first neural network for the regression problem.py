import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0

y_train = torch.sin(x_train)

plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('$y = sin(x)$');
plt.show()

noise = torch.randn(y_train.shape) / 5.

plt.plot(x_train.numpy(), noise.numpy(), 'o')
plt.axis([-10, 10, -1, 1])
plt.title('Gaussian noise');
plt.show()

y_train = y_train + noise
plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('noisy sin(x)')
plt.xlabel('x_train')
plt.ylabel('y_train');
plt.show()

x_train.unsqueeze_(1)  # Преобразуем строку в столбец [1, 2, 3, 4], станет так [1]
# [2]
# [3]
# [4]
y_train.unsqueeze_(1);

tmp = torch.Tensor([1, 2, 3])
print(tmp)
print(tmp.unsqueeze(1))
# tensor([1., 2., 3.])
# tensor([[1.],
#         [2.],
#         [3.]])


x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)
plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
plt.title('sin(x)')
plt.xlabel('x_validation')
plt.ylabel('y_validation');
plt.show()
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1);


# Мы применили метод unsqueeze_, так как хотим,
# чтобы каждый элемент был вектором (пусть и из одного числа)

# Задача регрессии – предсказание вещественного числа
# Метод unsqueeze_ добавляет тензору еще одну размерность

class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


sine_net = SineNet(50)


def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction');
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


predict(sine_net, x_validation, y_validation)

# task
# import torch
#
# class SineNet(torch.nn.Module):
#     def __init__(self, n_hidden_neurons):
#         super(SineNet, self).__init__()
#         self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
#         self.act1 = torch.nn.Tanh()
#         self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
#         self.act2 = torch.nn.Tanh()
#         self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.fc3(x)
#         return x
#
#
# sine_net = SineNet(int(input()))
# sine_net.forward(torch.tensor([1.]))
# print(sine_net)
#
# def predict(net, x, y):
#     y_pred = net.forward(x)
#
#     plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
#     plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction');
#     plt.legend(loc='upper left')
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.show()
# predict(sine_net, x_validation, y_validation)


optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)  # Параметры - веса


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


for epoch_index in range(2000):
    optimizer.zero_grad()

    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)

    loss_val.backward()

    optimizer.step()

predict(sine_net, x_validation, y_validation)

# task 2

import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (15.0, 5.0)


def target_function(x):
    return 2 ** x * torch.sin(2 ** -x)


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


net = RegressionNet(50)

x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


for epoch_index in range(1000):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()


# Функция оценки качества работы неросейти
def metric(pred, target):
    return (pred - target).abs().mean()


def predict(net, x, y):
    y_pred = net.forward(x)

    # Визуализация тестовых данных
    plt.plot(x.numpy(), y.numpy(), '-', label='Ground trurh')
    # Визуализация предсказания нейросети данных
    plt.plot(x.numpy(), y_pred.data.numpy(), 'x', c='g', label='Prediction')
    plt.title('2**x * torch.sin(2**-x)')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# Визуализация работы нейросети
predict(net, x_validation, y_validation)

# Проверка качества нейросети (погрешность)
print(metric(net.forward(x_validation), y_validation).item())
