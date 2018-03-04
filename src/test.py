import os
import glob 
from src.utils.data_utils import *
import pickle

def test_fer_model(img_folder, model = "nets/fer2013_net/pickled_net.p"):
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

    # append "/" to img_folder if it isn't present already
    if img_folder[-1] != '/':
        img_folder += "/"

    # get filenames and sort the in lexico-graphical order
    filenames =  glob.glob(img_folder + "*.jpg")
    filenames.sort()

    # process images into matric
    X_test = process_images(filenames)[0]
    
    # unpickle model
    model = load_pickle(open(model, "rb"))

    # get prediction vector
    preds = model.loss(X_test)

    return preds



