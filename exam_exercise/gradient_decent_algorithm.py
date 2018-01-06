import numpy as np

x_train = np.arange(1, 16, dtype=np.float32)
y_train = np.array([0.99, 1.97, 3.33, 4.11, 5.02, 6.20, 7.32, 8.00, 8.89, 10.01, 12.21, 11.76, 14.12, 12.74, 16.11])

learningrate = 0.01

We = np.random.rand()


class GradientDescent:
    def __init__(self, y, x):
        self.__t_train = y
        self.__x_train = x

    def cost_func(self, w):
        return np.sum((self.__t_train - (w*self.__x_train))**2)/len(self.__x_train)

    def numerical_derivative(self, f, x):
        delta_x = 1e-6

        return (f(x+delta_x) - f(x-delta_x))/(2*delta_x)

    def gradient_descent(self, f, init_W, learning_rate):
        w = init_W

        gradi = self.numerical_derivative(f, w)
        w = w - learning_rate*gradi
        return w

grad = GradientDescent(x_train, y_train)
print("initial W = ", We, ", initial cost = ", grad.cost_func(We), ", initial W = ", We, ", learning rate = ", learningrate)

for step in range(101):

    We = grad.gradient_descent(grad.cost_func, We, learningrate)

    if step%10 == 0:
        print("step = ", step, ", cost=", grad.cost_func(We), ", W=", We)

