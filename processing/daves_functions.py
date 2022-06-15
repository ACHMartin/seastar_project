# -*- coding: utf-8 -*-

def plotSimpleLine():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(-5,5,100)
    y = 2*x+1
    plt.plot(x, y, '-r', label='y=2x+1')
    plt.title('Graph of y=2x+1')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

