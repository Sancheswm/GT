import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

def SubGraph(s1, s2, loc):
    return int(str(s1) + str(s2) + str(loc))

def PlotOneLine(x, y, color=None, subId=None, ax=None):  # Added ax parameter
    if ax is None:
        if subId is not None:
            ax = plt.subplot(subId)
        else:
            ax = plt.gca() # Get current axes if no subplot is defined
    if color:
        ax.plot(x, y, color) # Use ax.plot
    else:
        ax.plot(x, y) # Use ax.plot


def PlotLineChart(x, y, xName='', yName='', subGraph=True):
    if x.shape != y.shape or len(x.shape) > 2:
        print(x.shape, y.shape)
        raise ValueError('Input Data Error for Plotter: Shapes must match and be 1D or 2D') # Raise ValueError

    fig = plt.figure(figsize=(8, 6)) # set figure size

    if subGraph:
        if len(x.shape) == 2:
            num_plots = x.shape[0]
            for i in range(num_plots):
                subId = SubGraph(num_plots, 1, i + 1)
                ax = fig.add_subplot(num_plots, 1, i + 1) # More flexible subplot creation
                PlotOneLine(x[i], y[i], ax=ax) # pass axes to PlotOneLine
                ax.set_xlabel(xName) # set x label for each subplot
                ax.set_ylabel(yName) # set y label for each subplot
                if i < num_plots -1: # remove duplicated x tick labels
                    ax.set_xticklabels([])
        else:
            ax = fig.add_subplot(1, 1, 1) # create subplot
            PlotOneLine(x, y, ax=ax) # pass axes to PlotOneLine
            ax.set_xlabel(xName) # set x label
            ax.set_ylabel(yName) # set y label
    else:
        if len(x.shape) == 2 and x.shape[0] > len(colors): # Corrected condition
            raise ValueError('Too Many Curves, Use SubGraph or provide more colors') # Raise ValueError
        elif len(x.shape) == 2:
            num_plots = x.shape[0]
            for i in range(num_plots):
                PlotOneLine(x[i], y[i], colors[i])
        else:
            PlotOneLine(x, y)
    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()
