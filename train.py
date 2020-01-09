#importing libraries
#To run this file insert following command in command propmt
#python train.py --train datasets/train --test datasets/val --model output/alexnet.hdf5
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from nn import lenet
from nn import minivggnet
from nn import alexnet
from nn import minigooglenet
from nn import googlenet
from nn import resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import argparse

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--train", required=True, help="path to train dataset")
#--train is the path to our plant village train dataset directory on disk.
ap.add_argument("-te", "--test", required=True, help="path to test dataset")
#--test is the path to our plant village train dataset directory on disk.
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


#model = lenet.LeNet.build(width=28, height=28, depth=3, classes=38)
#model = minivggnet.MiniVGGNet.build(width=32, height=32, depth=3, classes=38)
model = alexnet.AlexNet.build(width=227, height=227, depth=3, classes=38, reg=0.0002)
#model = minigooglenet.MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
#model = googlenet.GoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
#model = resnet.ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)

#image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, 
                                   zoom_range=0.15, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.15, 
                                   horizontal_flip=True, 
                                   fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
print("[INFO] loading images...")
train_data_dir = args["train"]     #directory of training data
test_data_dir = args["test"]      #directory of test data

training_set = train_datagen.flow_from_directory(train_data_dir, target_size=(227, 227),
	batch_size=batch_size, class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_data_dir, target_size=(227, 227),
	batch_size=batch_size, class_mode='categorical')

print(training_set.class_indices)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

print("[INFO] compiling model...")
opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit_generator(training_set, 
	steps_per_epoch=training_set.samples//batch_size, 
	validation_data=test_set,
	epochs=50,
	validation_steps=test_set.samples//batch_size,
    callbacks=[callbacks])

print("[Info] visualising model...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

print("[Info] saving model...")
model.save(args["model"])
