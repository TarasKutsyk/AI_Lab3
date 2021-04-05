from matplotlib import pyplot
from numpy import asarray
from sklearn import preprocessing, metrics
from tensorflow import keras
from tensorflow.keras import layers

from config import f, layersCount, neuronsPerLayerCount, inputRange

# задаємо множини визначення і значень функції
x = asarray([i for i in inputRange])
y = asarray([f(i) for i in x])

print(x.min(), x.max(), y.min(), y.max())

# переформовуємо означені множини у матриці
x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

# "стискаємо" отримані значення в область (0, 1)
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

print(x.min(), x.max(), y.min(), y.max())

# задаємо модель нейронної мережі
model = keras.Sequential(name = 'Approximator')

model.add(layers.Dense(neuronsPerLayerCount, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
for i in range(layersCount - 1):
    model.add(layers.Dense(neuronsPerLayerCount, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(1))

model.summary()

# означуємо функцію втрати і метод оптимізації
model.compile(loss='mse', optimizer='adam')

# починаємо навчання мережі, передаючи вхідні та бажені значення, і конфігурації навчання
model.fit(x, y, epochs=500, batch_size=10, verbose=0)

# отримуємо результати апроксимації
approximation = model.predict(x)

# повертаємо діапазони значень з (0, 1) до початкових
x_plot = scaler.inverse_transform(x)
y_plot = scaler.inverse_transform(y)
approximation_plot = scaler.inverse_transform(approximation)

# виводимо сумарну похибку
print('Mean squared error: %.4f' % metrics.mean_squared_error(y_plot, approximation_plot))

# виводимо результати у вигляді графіків
pyplot.scatter(x_plot, y_plot, label='Actual function')
pyplot.scatter(x_plot, approximation_plot, label='Approximation')

pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.legend()

pyplot.show()