import math
import random

import matplotlib.pyplot as plt


# генератор
class SemicircleGenerator(object):

    def __init__(self, StartLocation, RadiusList, Orientation):
        self.StartLocation = StartLocation
        self.MaxRadius = max(RadiusList)
        self.MinRadius = min(RadiusList)
        self.Orientation = Orientation

    def gen_semicircle_data(self, BatchSize):
        for _ in range(BatchSize):
            Radius = random.uniform(self.MinRadius, self.MaxRadius)
            BiasX = random.uniform(-Radius, Radius)
            BiasY = math.sqrt(Radius * Radius - BiasX * BiasX)
            if self.Orientation == "+":
                yield [BiasX + self.StartLocation[0], BiasY
                       + self.StartLocation[1]]
            else:
                yield [self.StartLocation[0] - BiasX,
                       self.StartLocation[1] - BiasY]


import numpy as np


class Neuron(object):

    def __init__(self, InputNum):
        self.Weight = np.zeros([InputNum + 1, 1])
        self.TrainRaito = 1

    def activation_function(self, InputData):
        return 1 if InputData > 0 else -1

    def feed_forward(self, InputData):
        return self.activation_function(
            np.matmul(self.Weight.T,
                      np.hstack((np.ones([1, 1]), InputData)).T))

    def train_one_step(self, InputData, RightResult):
        Result = self.feed_forward(InputData)
        if Result != RightResult:
            self.change_weight(InputData, RightResult, Result)

    def change_weight(self, InputData, RightResult, Result):
        self.Weight += self.TrainRaito * \
                       (RightResult - Result) * np.hstack(
            (np.ones([1, 1]), InputData)).T

    def set_train_ratio(self, Ratio):
        self.TrainRaito = Ratio


neural = Neuron(2)
dataset1 = SemicircleGenerator([0, 0], [4, 6], "+")
dataset2 = SemicircleGenerator([7, -2], [4, 6], "-")
testdata_x, testdata_y = [], []

for data in dataset1.gen_semicircle_data(1000):
    neural.train_one_step(np.array([data]), 1)
    testdata_x.append(data[0])
    testdata_y.append(data[1])
for data in dataset2.gen_semicircle_data(1000):
    neural.train_one_step(np.array([data]), -1)
    testdata_x.append(data[0])
    testdata_y.append(data[1])

x_data, y_data = [-6, 13], []

for i in x_data:  # выразили y
    y_data.append(- (neural.Weight[0][0] + i *
                     neural.Weight[1][0]) / (neural.Weight[2][0]))

plt.plot(x_data, y_data, color="red")
plt.scatter(testdata_x, testdata_y)
plt.show()
