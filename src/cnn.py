from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Activation, Flatten, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2
from src.utils.data_utils import *
from keras.preprocessing.image import ImageDataGenerator

# DATASET PARAMETERS
img_rows, img_cols = 48, 48
batch_size  = 128
nb_classes = 7
nb_epoch = 30
img_channel = 3

def get_new_model():
    
    # declare model
    model = Sequential()

    # FIRST LAYER
    model. add(Conv2D(64, (6, 6), activation = 'relu', data_format = "channels_first", input_shape = (1, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (2, 2)))

    # SECOND LAYER
    model.add(Conv2D(64, (6, 6), activation = 'relu'))
    model.add(Conv2D(64, (6, 6), activation = 'relu'))
    model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))

    # THIRD LAYER
    model.add(Conv2D(64, (6, 6), activation = 'relu'))
    model.add(Conv2D(64, (6, 6), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    # FCC
    model.add(Dense(1003, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1003, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes, activation = 'softmax'))
        
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(),
                  metrics = ['accuracy'])
    
    
    model.summary()

    return model





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

# create generators
gen = ImageDataGenerator()

train_generator = gen.flow(X_train, y_train, batch_size = batch_size)

# get a new model
model = get_new_model()

model.fit_generator(train_generator,
                    steps_per_epoch = batch_size,
                    epochs = nb_epoch,
                    verbose = 1)
                    

