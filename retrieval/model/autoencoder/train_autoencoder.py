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
import os


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


def ae_preprocess(img_path, input_shape=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    if input_shape is not None:
        img = tf.image.resize(img, input_shape[:2])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return (img, img)


# construct the argument parse and parse the arguments
RUN_FOLDER = 'output'

# path to output reconstruction visualization file
VIS = "output/recon_vis.png"
# path to output plot file
PLOT = "output/plot.png"

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 0.0005
BATCH_SIZE = 100
input_shape = (224, 224, 3)

# load the MNIST dataset
print("[INFO] loading dataset...")

image_dataset_path = "C:/Dev/ImageData/t/**/*.jpg"

fnames = glob.glob(image_dataset_path, recursive=True)
list_ds = tf.data.Dataset.from_tensor_slices(fnames)
ds = list_ds.map(lambda x: ae_preprocess(x, input_shape), num_parallel_calls=-1)
dataset = ds.batch(BATCH_SIZE).shuffle(len(fnames)).prefetch(-1)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
AE = Autoencoder(
    input_dim=(224, 224, 3),
    encoder_conv_filters=[32, 64, 64, 128, 256, 1024],
    encoder_conv_kernel_size=[3, 3, 3, 3, 3, 1],
    encoder_conv_strides=[2, 2, 2, 2, 2, 1],
    decoder_conv_t_filters=[256, 128, 64, 64, 32, 3],
    decoder_conv_t_kernel_size=[1, 3, 3, 3, 3, 3],
    decoder_conv_t_strides=[1, 2, 2, 2, 2, 2],
    z_dim=1000,
    use_batch_norm=True
)

if os.path.exists(os.path.join(RUN_FOLDER, 'weights/weights.h5')):
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

AE.compile(INIT_LR, BATCH_SIZE)

# train the convolutional autoencoder
H = AE.train(
    dataset,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER
)
# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
img_dataset = []
for fname in fnames[-10:]:
    img_dataset.append(ae_preprocess(fname, input_shape)[0])
img_dataset = np.array(img_dataset)

decoded = AE.model.predict(img_dataset)
vis = visualize_predictions(decoded, img_dataset)
cv2.imwrite(VIS, vis)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT)
