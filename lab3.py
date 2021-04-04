from config import f, layersCount

# example of creating a univariate dataset with a given mapping function
from matplotlib import pyplot
# define the input data
myRange = [i for i in range(0,30)]
# define the output data
y = [f(i) for i in myRange]
# plot the input versus the output
pyplot.scatter(myRange,y)
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.show()