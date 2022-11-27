# for plotting bar graphs
import pandas
import matplotlib.pyplot as plot

def graph(accuracy, feature_set, title):
    # A python dictionary
    data = { "Current Feature Set":feature_set, "Accuracy": accuracy }
    # load data into dataframe
    dataFrame = pandas.DataFrame(data=data)
    # plot bar chart
    dataFrame.plot.bar(x='Current Feature Set', y='Accuracy', title=title)
    plot.show(block=True)
    return