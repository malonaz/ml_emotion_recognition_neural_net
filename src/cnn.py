from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
from src.utils.data_utils import *
from keras.preprocessing.image import ImageDataGenerator

def get_new_model():
    
    # declare model
    model = Sequential()

    ### ADD LAYERS
    # add a 2d convolution layer, with valid padding. make sure channels come first
    model.add(Convolution2D(kernel_size = 5, strides = 1, data_format = "channels_first",
                        filters = 64, padding = 'valid', input_shape = (1, 48, 48)))
    
    # add a relu layer. input will be shape = (None, 64, 44, 44)
    model.add(Dense(units = 512, activation='relu', use_bias = False))
    
    # add a 2 x 2 max pool layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # add an affine + relu
    model.add(Dense(units = 512, activation = 'relu', use_bias = True))

    # flatten then add output layer with softmax activaiton
    model.add(Flatten())
    model.add(Dense(7, activation = 'softmax'))


    
    # define optimizer
    sgd = SGD(lr = 0.01,
              momentum = 0.9,
              decay = 0.95)
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = sgd,
                  metrics = ['accuracy'])
    
    
    model.summary()

    return model



img_rows, img_cols = 48, 48
batch_size  = 128
nb_classes = 7
nb_epoch = 1200
img_channel = 3


# get data
data = load_data()

# split the data
X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]

# Change x values to to greyscale using the average method
X_train,  X_val  = X_train.mean(axis = 1), X_val.mean(axis = 1)

# reshape matrices
X_train = X_train.reshape(X_train.shape[0], 1,  img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)


# convert to binary arrays
y_train = np_utils.to_categorical(y_train, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)


# get a new model
model = get_new_model()

model.fit(X_train,
          y_train,
          epochs = 5,
          batch_size = 32,
          validation_data = (X_val, y_val),
          verbose = 1)

