import matplotlib.pyplot as plt
import numpy as np
import random
import math


class SemicircleGenerator(object):

    def __init__(self, StartLocation, RadiusList, Orientation):
        super(SemicircleGenerator, self).__init__()
        self.StartLocation = StartLocation
        self.MaxRadius = max(RadiusList)
        self.MinRadius = min(RadiusList)
        self.Orientation = Orientation

    def Gen_SemicircleData(self, BatchSize):
        for _ in range(BatchSize):
            Radius = random.uniform(self.MinRadius, self.MaxRadius)
            BiasX = random.uniform(- Radius, Radius)
            BiasY = math.sqrt(Radius * Radius - BiasX * BiasX)
            if self.Orientation == "+":
                yield [BiasX + self.StartLocation[0], BiasY
                       + self.StartLocation[1]]
            else:
                yield [self.StartLocation[0] - BiasX, self.StartLocation[1] - BiasY]

dataset1 = SemicircleGenerator([0, 0], [4, 6], "+")
dataset2 = SemicircleGenerator([7, -2], [4, 6], "-")
print(dataset1.__dict__)