from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam, sgd
from CONFIG import PATCH_SIZE

def generate_model():

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(PATCH_SIZE, PATCH_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    adam = sgd(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mse', 'accuracy'])

    return model