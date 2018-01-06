import numpy as np


class Forward_Propagation:

    def __init__(self):
        self.__x_train = np.array([0.8, 0.1], ndmin=2).reshape(2, 1)
        self.__y_train = np.array([1.0, 1.0], ndmin=2).reshape(2, 1)
        self.__w1 = None; self.__w2 = None; self.__w3 = None; self.__w4 = None
        self.__e1 = None; self.__e2 = None; self.__e3 = None; self.__e4 = None
        self.__Y1 = None; self.__Y2 = None; self.__Y3 = None;
        self.__final = None


    def __init_weight(self):
        self.__w1 = np.random.rand(3, 2)
        self.__w2 = np.random.rand(2, 3)
        self.__w3 = np.random.rand(3, 2)
        self.__w4 = np.random.rand(2, 3)

    @staticmethod
    def __sigmoid(x):
        return 1 / (1+np.exp(-x))

    def forward_propagation(self):
        self.__init_weight()
        print(np.shape(self.__x_train), np.shape(self.__w1))
        h1_y = np.dot(self.__w1, self.__x_train)
        self.__Y1 = self.__sigmoid(h1_y)

        h2_y = np.dot(self.__w2, self.__Y1)
        self.__Y2 = self.__sigmoid(h2_y)

        h3_y = np.dot(self.__w3, self.__Y2)
        self.__Y3 = self.__sigmoid(h3_y)
        print(np.shape(self.__Y3))
        h4_y = np.dot(self.__w4, self.__Y3)
        self.__final = self.__sigmoid(h4_y)

        return self.__final

    def back_propagation(self):
        self.__e1 = self.__y_train - self.__final
        self.__e2 = np.dot(self.__w4.T, self.__e1)
        self.__e3 = np.dot(self.__w3.T, self.__e2)
        self.__e4 = np.dot(self.__w2.T, self.__e3)

        print(np.shape(self.__e1), np.shape(self.__e2), np.shape(self.__e3), np.shape(self.__e4))
        return self.__e1

    def update(self):
        learning_rate = 0.01
        print(self.__w4, self.__w3, self.__w2, self.__w1)
        self.__w4 = self.__w4 + learning_rate * np.dot(self.__e1 * self.__final * (1.0 - self.__final), self.__Y3.T)
        self.__w3 = self.__w3 + learning_rate * np.dot(self.__e2 * self.__Y3 * (1 - self.__Y3), self.__Y2.T)
        self.__w2 = self.__w2 + learning_rate * np.dot(self.__e3 * self.__Y2 * (1 - self.__Y2), self.__Y1.T)
        self.__w1 = self.__w1 + learning_rate * np.dot(self.__e4 * self.__Y1 * (1 - self.__Y1), self.__x_train.T)
        print(self.__w4, self.__w3, self.__w2, self.__w1)

fp = Forward_Propagation()
for i in range(100):
    print("forward")
    print(fp.forward_propagation())
    print("back")
    print(fp.back_propagation())
    print("update")
    fp.update()