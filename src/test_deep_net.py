import glob 
from src.utils.data_utils import *
from keras.models import load_model
from keras.utils import np_utils


data = load_data()

def test_fer_model(img_folder, model_filename = "nets/deep_net/cnn_model.h5"):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the filename of the images) and your best model to predict
    the facial expression of each image.

    args:
    - image_folder: Path to the images to be tested
    - model: Path to the h5py file of the model

    returns:
    - preds: A numpy vector of size N with N being the number of images in img_folder.
    """

    ### GET FILENAMES
    # append "/" to img_folder if it isn't present already
    if img_folder[-1] != '/':
        img_folder += "/"

    # get filenames and sort the in lexico-graphical order
    filenames =  glob.glob(img_folder + "*.jpg")
    filenames.sort()


    
    ### PROCESS IMAGES
    # unpickle mean image
    mean_image = pickle.load(open("nets/optimal_net/mean_image_fer2013.p", "rb"))

    # process images and subtract mean image
    X_test = process_images(filenames)[0] - mean_image

    # transpose so channels come first
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # average to gray scale and reshape
    X_test = X_test.mean(axis = 1)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)

    ### TEST MODEL
    # load model
    model = load_model(model_filename)
    
    # get model's score on X 
    scores = model.predict(x = X_test)
    
    # get prediction vector using argmax. They need a mapping correction because of keras indexing issue
    preds = scores.argmax(axis = 1)
    mapping = [1, 2, 3, 4, 5, 6, 0]
    for i in range(len(preds)):
        preds[i] = mapping[preds[i]]
        
    
    return preds

