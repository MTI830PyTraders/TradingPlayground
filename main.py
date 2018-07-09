import download_data
import numpy as np
import pandas as pd

BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'
TICKER = 'TSLA'

train_data = download_data.get_training_dataset(BEGINNING_DATE, ENDING_DATE, TICKER)


train_x = np.array(train_data[['Sentiment', 'close']])
train_y = np.array(train_data['fcf'])

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(2, )),
    Activation('relu'),
    Dense(1),
    Activation('softmax'),
])

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, batch_size=32
          # ,validation_split=0.10
          )

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

print('saved model!')
