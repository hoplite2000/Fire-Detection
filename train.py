import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from firedetectnet import FireDetectNet
from lrfinder import LearningRateFinder
import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys

def load_dataset(dataset_path):
    imagePaths = list(paths.list_images(dataset_path))
    data = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (128, 128))
        data.append(image)
    return np.array(data, dtype="float32")

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lrfind", type=int, default=0,	help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

print("[INFO] loading data...")
fireData = load_dataset(config.Fire_path)
nonFireData = load_dataset(config.Non_fire_path)

fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

labels = to_categorical(labels, num_classes=2)
#classTotals = labels.sum(axis=0)
#classWeight = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=config.Test_split, random_state=42)

aug = ImageDataGenerator(rotation_range=30,	zoom_range=0.15, width_shift_range=0.2,	height_shift_range=0.2,	shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=config.Init_lr, decay=config.Init_lr / config.Epochs)
#opt = SGD(lr=config.Init_lr, momentum=0.9,decay=config.Init_lr / config.Epochs)
model = FireDetectNet.build(width=128, height=128, depth=3, classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

if args["lrfind"] > 0:
    print("[INFO] finding learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config.Batch_size), 1e-10, 1e+1, stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.Batch_size))), epochs=20, batchSize=config.Batch_size)
    lrf.plot_loss()
    plt.savefig(config.Lr_find_plot_path)
    print("[INFO] learning rate finder complete")
    print("[INFO] examine plot and adjust learning rates before training")

print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=config.Batch_size), validation_data=(testX, testY), steps_per_epoch=trainX.shape[0] // config.Batch_size, epochs=config.Epochs, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=config.Batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.Classes))

print("[INFO] serializing network to '{}'...".format(config.Model_path))
model.save(config.Model_path)

N = np.arange(0, config.Epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.Training_plot_path)