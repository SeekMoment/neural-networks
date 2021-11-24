
import torch

x = torch.tensor([[1., 2., 3., 4.],
                  [5., 6., 7., 8.],
                  [9., 10., 11., 12.]], requires_grad=True)
# requires_grad=True
# это указывает на то что впоследтствии будут браться производные
# Сообщает о том, что данный тензор является переменной, по которой нужно будет считать градиенты
#  Превращает тензор-константу в тензор-переменную

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

function = 10 * (x ** 2).sum()
function.backward()
print(x.grad, '<- gradient')
# Результат операции backward хранится в x.grad


# выводим порядок операций
# print(function.grad_fn)
# print(function.grad_fn.next_functions[0][0])
# print(function.grad_fn.next_functions[0][0].next_functions[0][0])
# print(function.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])


# task 1

# w = torch.tensor([[5., 10.],
#                   [1., 2.]], requires_grad=True)
#
# function = torch.log(torch.log(w + 7)).prod()
#
# function.backward()


# x = torch.tensor([[1., 2., 3., 4.],
#                   [5., 6., 7., 8.],
#                   [9., 10., 11., 12.]], requires_grad=True)
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# x = x.to(device)
#
# function = 10 * (x ** 2).sum()
# function.backward()
#
# # обновление тензора х
# x.data -= 0.001 * x.grad  # тот же тензор, только с requires_grad = false
# x.grad.zero_()  # градиент всегда накапливается, поэтому его нужно обнулять


# task 2

# w = torch.tensor([[5., 10.],
#                   [1., 2.]], requires_grad=True)
# alpha = 0.001
# for i in range(500):
#     function = torch.log(torch.log(w + 7)).prod()
#     function.backward()
#     w.data = w.data - alpha * w.grad
#     w.grad.zero_()
#
# print(w)



# x = torch.tensor([8., 8.], requires_grad=True)
# def function_parabola(variable):
#     return 10 * (variable ** 2).sum()
#
#
# def make_gradient_step(function, varibale):
#     function_result = function(varibale)
#     function_result.backward()
#     varibale.data -= 0.001 * varibale.grad
#     varibale.grad.zero_()
#
# for i in range(500):
#     make_gradient_step(function_parabola, x)




# x = torch.tensor([8., 8.], requires_grad=True)
#
# optimizer = torch.optim.SGD([x], lr=0.001)
#
# def function_parabola(variable):
#     return 10 * (variable ** 2).sum()
#
#
# def make_gradient_step(function, varibale):
#     function_result = function(varibale)
#     function_result.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
#
# for i in range(500):
#     make_gradient_step(function_parabola, x)



# task 3

w = torch.tensor([[5., 10.],
                  [1., 2.]], requires_grad=True)
optimizer = torch.optim.SGD([w], lr = 0.001)
for i in range(500):
    function = torch.log(torch.log(w + 7)).prod()
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w)
