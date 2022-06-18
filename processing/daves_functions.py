# -*- coding: utf-8 -*-
# Comment
def plotSimpleLine():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(-5,5,200)
    y = 3*x+1
    plt.plot(x, y, '-b', label='y=2x+1')
    plt.title('Graph of y=2x+1')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

