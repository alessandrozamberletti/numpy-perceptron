import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        np.random.seed(1)
        self.weights = self.__random_init(2)
        self.bias = self.__random_init(1)

    @staticmethod
    def __random_init(size):
        return 2 * np.random.rand(size) - 1

    @staticmethod
    def __transfer(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __prediction(x):
        return 1 if x >= 0.5 else 0

    def __activation(self, x):
        return self.__prediction(self.__transfer(np.dot(self.weights, x) + self.bias))

    @staticmethod
    def __plot_loss(loss_plot, mse):
        loss_plot.clear()
        loss_plot.set_title('Train Loss')
        loss_plot.set_xlabel('Epoch')
        loss_plot.set_ylabel('MSE')

        loss_plot.plot(mse)
        plt.pause(.0001)

    def __plot_rule(self, rule_plot, training):
        rule_plot.clear()
        rule_plot.set_title('Decision Rule')

        space = np.arange(np.min(training)[0]-.5, np.max(training)[0]+1)
        x_grid, y_grid = np.meshgrid(space, space)
        z_grid = self.__transfer(x_grid * self.weights[0] + y_grid * self.weights[1] + self.bias)
        rule_plot.contourf(x_grid, y_grid, z_grid, levels=[0, 0.5, 1], alpha=.5)

        for observation, expectation in training:
            prediction = self.__activation(observation)
            c = 'green' if prediction == expectation[0] else 'red'
            plt.plot(observation[0], observation[1], marker='v', markersize=10, color=c)
            rule_plot.annotate('{0}$\mapsto${1}'.format(observation, prediction), xy=np.add(observation, -.1))

    def fit(self, samples, epochs=100, info=True):
        if info:
            plt.ion()
            loss_plot = plt.subplot(121)
            rule_plot = plt.subplot(122)

        mse = []
        for epoch in range(epochs):
            square_losses = []
            for sample in np.random.permutation(samples):
                observation, expectation = sample
                prediction = self.__transfer(np.dot(self.weights, observation) + self.bias)
                loss = expectation - prediction
                square_losses.append(loss**2)
                self.weights += loss * observation
                self.bias += loss

            if info:
                mse.append(np.average(square_losses))
                self.__plot_loss(loss_plot, mse)
                self.__plot_rule(rule_plot, samples)
                print('Epoch: {0}\tMSE:\t{1:.13f}'.format(epoch, mse[-1]))


if __name__ == "__main__":
    data = [([0, 0], [0]),
            ([0, 1], [0]),
            ([1, 1], [1]),
            ([1, 0], [0])]

    p = Perceptron()
    p.fit(data)
