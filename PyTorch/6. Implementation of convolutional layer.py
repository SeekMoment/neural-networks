# Вспомним как работает сверточный слой:
#
#     на вход подается массив изображений, еще он называется батчем
#
#     к каждому изображению по границам добавляются нули
#
#     по каждому изображению "скользит" каждый из фильтров сверточного слоя

# task  number 1
# import torch
# import torch.nn.functional as F
# # Создаем входной массив из двух изображений RGB 3*3
# input_images = torch.tensor(
#       [[[[0,  1,  2],
#          [3,  4,  5],
#          [6,  7,  8]],
#
#         [[9, 10, 11],
#          [12, 13, 14],
#          [15, 16, 17]],
#
#         [[18, 19, 20],
#          [21, 22, 23],
#          [24, 25, 26]]],
#
#
#        [[[27, 28, 29],
#          [30, 31, 32],
#          [33, 34, 35]],
#
#         [[36, 37, 38],
#          [39, 40, 41],
#          [42, 43, 44]],
#
#         [[45, 46, 47],
#          [48, 49, 50],
#          [51, 52, 53]]]])
#
#
# def get_padding2d(input_images):
#     padded_images = F.pad(input=input_images, pad=(1, 1, 1, 1), mode='constant', value=0)
#     return padded_images.float()
#
#
# correct_padded_images = torch.tensor(
#        [[[[0.,  0.,  0.,  0.,  0.],
#           [0.,  0.,  1.,  2.,  0.],
#           [0.,  3.,  4.,  5.,  0.],
#           [0.,  6.,  7.,  8.,  0.],
#           [0.,  0.,  0.,  0.,  0.]],
#
#          [[0.,  0.,  0.,  0.,  0.],
#           [0.,  9., 10., 11.,  0.],
#           [0., 12., 13., 14.,  0.],
#           [0., 15., 16., 17.,  0.],
#           [0.,  0.,  0.,  0.,  0.]],
#
#          [[0.,  0.,  0.,  0.,  0.],
#           [0., 18., 19., 20.,  0.],
#           [0., 21., 22., 23.,  0.],
#           [0., 24., 25., 26.,  0.],
#           [0.,  0.,  0.,  0.,  0.]]],
#
#
#         [[[0.,  0.,  0.,  0.,  0.],
#           [0., 27., 28., 29.,  0.],
#           [0., 30., 31., 32.,  0.],
#           [0., 33., 34., 35.,  0.],
#           [0.,  0.,  0.,  0.,  0.]],
#
#          [[0.,  0.,  0.,  0.,  0.],
#           [0., 36., 37., 38.,  0.],
#           [0., 39., 40., 41.,  0.],
#           [0., 42., 43., 44.,  0.],
#           [0.,  0.,  0.,  0.,  0.]],
#
#          [[0.,  0.,  0.,  0.,  0.],
#           [0., 45., 46., 47.,  0.],
#           [0., 48., 49., 50.,  0.],
#           [0., 51., 52., 53.,  0.],
#           [0.,  0.,  0.,  0.,  0.]]]])
#
# print(torch.allclose(get_padding2d(input_images), correct_padded_images))


# task number 2

# Каждый фильтр имеет следующую размерность:
#
#     число слоев во входном изображении (для RGB это 3)
#
#     высота фильтра
#
#     ширина фильтра
#
# В ядре (кернеле) все фильтры имеют одинаковые размерность,
# поэтому ширину и высоту фильтров называют шириной и высотой ядра.
# Чаще всего ширина ядра равна высоте ядра, в таком случае их
# называют размером ядра (kernel_size).


# import numpy as np
# import torch
#
# # Входная размерность (число изображений в батче, число слоев в одном изображении, высота и ширина)
# # Количество фильтров
# # Размер фильтров (считаем что ширина совпадает с шириной)
# # stride
# # padding
# def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
#     out_shape = torch.tensor([input_matrix_shape[0], out_channels,
#                               int(np.floor((input_matrix_shape[2] - (kernel_size) + 2 * padding) // stride) + 1),
#                               int(np.floor((input_matrix_shape[3] - (kernel_size) + 2 * padding) // stride) + 1)])
#
#     return out_shape
#
# print(np.array_equal(
#     calc_out_shape(input_matrix_shape=[2, 3, 5, 5],
#                    out_channels=10,
#                    kernel_size=3,
#                    stride=2,
#                    padding=1),
#     [2, 10, 8, 8]))
# print(calc_out_shape(input_matrix_shape=[2, 3, 5, 5],
#                    out_channels=10,
#                    kernel_size=3,
#                    stride=2,
#                    padding=0))
# print()
#


# import torch
# from abc import ABC, abstractmethod
#
#
# # абстрактный класс для сверточного слоя
# class ABCConv2d(ABC):
#     def __int__(self, in_channels, out_channels, kernel_size, stride):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#     def set_kernel(self, kernel):
#         self.kernel = kernel
#
#     @abstractmethod
#     def __call__(self, input_tensor):
#         pass
#
#
# # класс-обертка над torch.nn.Conv2d для унификации интерфейса
# class Conv2d(ABCConv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=False)
#
#     def set_kernel(self, kernel):
#         self.conv2d.weight.data = kernel
#
#     def __call__(self, input_tensor):
#         return self.conv2d(input_tensor)
#
#
# # функция, создающая объект класса cls и возвращающая свертку от input_matrix
# def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
#     out_channels = kernel.shape[0]
#     in_channels = kernel.shape[1]
#     kernel_size = kernel.shape[2]
#
#     layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
#     layer.set_kernel(kernel)
#
#     return layer(input_matrix)
#
#
# # Функция, тестирующая класс conv2d_cls.
# # Возвращает True, если свертка совпадает со сверткой с помощью torch.nn.Conv2d.
# def test_conv2d_layer(conv2d_layer_class, batch_size=2,
#                       input_height=4, input_width=4, stride=2):
#     kernel = torch.tensor(
#         [[[[0., 1, 0],
#            [1, 2, 1],
#            [0, 1, 0]],
#
#           [[1, 2, 1],
#            [0, 3, 3],
#            [0, 1, 10]],
#
#           [[10, 11, 12],
#            [13, 14, 15],
#            [16, 17, 18]]]])
#
#     in_channels = kernel.shape[1]
#
#     input_tensor = torch.arange(0, batch_size * in_channels *
#                                 input_height * input_width,
#                                 out=torch.FloatTensor()) \
#         .reshape(batch_size, in_channels, input_height, input_width)
#
#     custom_conv2d_out = create_and_call_conv2d_layer(
#         conv2d_layer_class, stride, kernel, input_tensor)
#     conv2d_out = create_and_call_conv2d_layer(
#         Conv2d, stride, kernel, input_tensor)
#
#     return torch.allclose(custom_conv2d_out, conv2d_out) and (custom_conv2d_out.shape == conv2d_out.shape)
#
#
# print(test_conv2d_layer(Conv2d))


# Task number 3


# import torch
# from abc import ABC, abstractmethod
#
#
# def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
#     batch_size, channels_count, input_height, input_width = input_matrix_shape
#     output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#     output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#
#     return batch_size, out_channels, output_height, output_width
#
#
# class ABCConv2d(ABC):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#     def set_kernel(self, kernel):
#         self.kernel = kernel
#
#     @abstractmethod
#     def __call__(self, input_tensor):
#         pass
#
#
# class Conv2d(ABCConv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                                       stride, padding=0, bias=False)
#
#     def set_kernel(self, kernel):
#         self.conv2d.weight.data = kernel
#
#     def __call__(self, input_tensor):
#         return self.conv2d(input_tensor)
#
#
# def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
#     out_channels = kernel.shape[0]
#     in_channels = kernel.shape[1]
#     kernel_size = kernel.shape[2]
#
#     layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
#     layer.set_kernel(kernel)
#
#     return layer(input_matrix)
#
#
# def test_conv2d_layer(conv2d_layer_class, batch_size=2,
#                       input_height=4, input_width=4, stride=2):
#     kernel = torch.tensor(
#         [[[[0., 1, 0],
#            [1, 2, 1],
#            [0, 1, 0]],
#
#           [[1, 2, 1],
#            [0, 3, 3],
#            [0, 1, 10]],
#
#           [[10, 11, 12],
#            [13, 14, 15],
#            [16, 17, 18]]]])
#
#     in_channels = kernel.shape[1]
#
#     input_tensor = torch.arange(0, batch_size * in_channels *
#                                 input_height * input_width,
#                                 out=torch.FloatTensor()) \
#         .reshape(batch_size, in_channels, input_height, input_width)
#
#     custom_conv2d_out = create_and_call_conv2d_layer(
#         conv2d_layer_class, stride, kernel, input_tensor)
#     conv2d_out = create_and_call_conv2d_layer(
#         Conv2d, stride, kernel, input_tensor)
#
#     return torch.allclose(custom_conv2d_out, conv2d_out) \
#            and (custom_conv2d_out.shape == conv2d_out.shape)
#
#
# # Сверточный слой через циклы.
# class Conv2dLoop(ABCConv2d):
#     def __call__(self, input_tensor):
#         batch_size, out_channels, output_height, output_width = calc_out_shape(
#             input_tensor.shape,
#             self.out_channels,
#             self.kernel_size,
#             self.stride,
#             padding=0)
#
#         # создадим выходной тензор, заполненный нулями
#         output_tensor = torch.zeros(batch_size, out_channels, output_height, output_width)
#         # вычисление свертки с использованием циклов.
#         # цикл по входным батчам(изображениям)
#         for num_batch, batch in enumerate(input_tensor):
#
#             # цикл по фильтрам (количество фильтров совпадает с количеством выходных каналов)
#             for num_kernel, kernel in enumerate(self.kernel):
#
#                 # цикл по размерам выходного изображения
#                 for i in range(output_height):
#                     for j in range(output_width):
#                         # вырезаем кусочек из батча (сразу по всем входным каналам)
#                         current_row = self.stride * i
#                         current_column = self.stride * j
#                         current_slice = batch[:, current_row:current_row + self.kernel_size,
#                                         current_column:current_column + self.kernel_size]
#
#                         # умножаем кусочек на фильтр
#                         res = float((current_slice * kernel).sum())
#
#                         # заполняем ячейку в выходном тензоре
#                         output_tensor[num_batch, num_kernel, i, j] = res
#
#         return output_tensor
#
#
# print(test_conv2d_layer(Conv2dLoop))


# TASK NUMBER 4

# import torch
# from abc import ABC, abstractmethod
#
#
# def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
#     batch_size, channels_count, input_height, input_width = input_matrix_shape
#     output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#     output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#
#     return batch_size, out_channels, output_height, output_width
#
#
# class ABCConv2d(ABC):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#     def set_kernel(self, kernel):
#         self.kernel = kernel
#
#     @abstractmethod
#     def __call__(self, input_tensor):
#         pass
#
#
# class Conv2d(ABCConv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                                       stride, padding=0, bias=False)
#
#     def set_kernel(self, kernel):
#         self.conv2d.weight.data = kernel
#
#     def __call__(self, input_tensor):
#         return self.conv2d(input_tensor)
#
#
# def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
#     out_channels = kernel.shape[0]
#     in_channels = kernel.shape[1]
#     kernel_size = kernel.shape[2]
#
#     layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
#     layer.set_kernel(kernel)
#
#     return layer(input_matrix)
#
#
# def test_conv2d_layer(conv2d_layer_class, batch_size=2,
#                       input_height=4, input_width=4, stride=2):
#     kernel = torch.tensor(
#         [[[[0., 1, 0],
#            [1, 2, 1],
#            [0, 1, 0]],
#
#           [[1, 2, 1],
#            [0, 3, 3],
#            [0, 1, 10]],
#
#           [[10, 11, 12],
#            [13, 14, 15],
#            [16, 17, 18]]]])
#
#     in_channels = kernel.shape[1]
#
#     input_tensor = torch.arange(0, batch_size * in_channels *
#                                 input_height * input_width,
#                                 out=torch.FloatTensor()) \
#         .reshape(batch_size, in_channels, input_height, input_width)
#
#     custom_conv2d_out = create_and_call_conv2d_layer(
#         conv2d_layer_class, stride, kernel, input_tensor)
#     conv2d_out = create_and_call_conv2d_layer(
#         Conv2d, stride, kernel, input_tensor)
#
#     return torch.allclose(custom_conv2d_out, conv2d_out) \
#            and (custom_conv2d_out.shape == conv2d_out.shape)
#
#
# class Conv2dMatrix(ABCConv2d):
#     # Функция преобразование кернела в матрицу нужного вида.
#     def _unsqueeze_kernel(self, torch_input, output_height, output_width):
#         m = torch.nn.ZeroPad2d((0, 1, 0, 1))
#         kernel_unsqueezed = m(self.kernel)
#         kernel_unsqueezed = kernel_unsqueezed.reshape(1, -1)
#         return kernel_unsqueezed
#
#     def __call__(self, torch_input):
#         batch_size, out_channels, output_height, output_width \
#             = calc_out_shape(
#             input_matrix_shape=torch_input.shape,
#             out_channels=self.kernel.shape[0],
#             kernel_size=self.kernel.shape[2],
#             stride=self.stride,
#             padding=0)
#
#         kernel_unsqueezed = self._unsqueeze_kernel(torch_input, output_height, output_width)
#         result = kernel_unsqueezed @ torch_input.view((batch_size, -1)).permute(1, 0)
#         return result.permute(1, 0).view((batch_size, self.out_channels,
#                                           output_height, output_width))
#
#
# print(test_conv2d_layer(Conv2dMatrix))




# TASK NUMBER 5

import torch
from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrixV2(ABCConv2d):
    # Функция преобразования кернела в нужный формат.
    def _convert_kernel(self):
        converted_kernel = self.kernel.view(self.kernel.shape[0], -1) # Реализуйте преобразование кернела.
        return converted_kernel

    # Функция преобразования входа в нужный формат.
    def _convert_input(self, torch_input, output_height, output_width):
        converted_input = torch.zeros(torch_input.shape[0],
                                      output_height * output_width,
                                      torch_input.shape[1],
                                      self.kernel_size,
                                      self.kernel_size)

        for row in range(output_height):
            for col in range(output_width):
                converted_input[:, row * output_width + col, :, :, :] = \
                    torch_input[:, :, row * self.stride:row * self.stride + self.kernel_size,
                    col * self.stride:col * self.stride + self.kernel_size]

        converted_input = converted_input.view(torch_input.shape[0] * output_height * output_width, -1)
        converted_input.t_()  # Реализуйте преобразование входа.
        return converted_input

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)

        conv2d_out_alternative_matrix_v2 = converted_kernel @ converted_input
        return conv2d_out_alternative_matrix_v2.transpose(0, 1).view(torch_input.shape[0],
                                                     self.out_channels, output_height,
                                                     output_width).transpose(1, 3).transpose(2, 3)

print(test_conv2d_layer(Conv2dMatrixV2))
