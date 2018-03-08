import glob 
from src.utils.data_utils import *
import pickle


def test_fer_model(img_folder, model_filename = "nets/optimal_net/pickled_net.p"):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the filename of the images) and your best model to predict
    the facial expression of each image.

    args:
    - image_folder: Path to the images to be tested
    - model: Path to the pickled model

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



    ### TEST MODEL
    # unpickle model
    model = load_pickle(open(model_filename, "rb"))

    # get model's score on X 
    scores = model.loss(X_test)

    # get prediction vector using argmax
    preds = scores.argmax(axis = 1)

    
    return preds
