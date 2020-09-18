import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from guide.model.autoencoder.AE import Autoencoder

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2


def visualize_predictions(decoded, gt, samples=10):
    # initialize our list of output images
    outputs = None

    # loop over our number of output samples
    for i in range(0, samples):
        # grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output
        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

        # return the output images
        return outputs

def preprocess(img_path, input_shape=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    if input_shape is not None:
        img = tf.image.resize(img, input_shape[:2])
    img = img / 255.0
    return img

# construct the argument parse and parse the arguments

# path to output trained autoencoder
MODEL = "output/autoencoder.h5"
# path to output reconstruction visualization file
VIS = "output/recon_vis.png"
# path to output plot file
PLOT = "output/plot.png"

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
input_shape = (224, 224, 3)

# load the MNIST dataset
print("[INFO] loading dataset...")

image_dataset = "../../../image/train_load/*.jpg"

fnames = glob.glob(image_dataset, recursive=True)
list_ds = tf.data.Dataset.from_tensor_slices(fnames)
ds = list_ds.map(lambda x: preprocess(x, input_shape), num_parallel_calls=-1)
dataset = ds.batch(BS).prefetch(-1)


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

# train the convolutional autoencoder
H = autoencoder.fit(dataset, dataset,
                    validation_data=(testX, testX),
                    epochs=EPOCHS,
                    batch_size=BS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite(VIS, vis)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT)

# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save(MODEL, save_format="h5")