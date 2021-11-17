import numpy as np
import pandas as pd 
import os
import glob
import random
from skimage import io
import skimage.io as io
import skimage.transform as trans
import seaborn as sns
import zipfile
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, normalize
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import MaxPooling2D,Activation, BatchNormalization,Conv2D,concatenate, Conv2DTranspose
from tensorflow.keras import datasets,layers
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

from data_preProcessing import *
from model import *

smooth=1e-6
def dice_loss(y_true, y_pred):
    y_truef=keras.flatten(y_true)
    y_predf=keras.flatten(y_pred)
    And=keras.sum(y_truef* y_predf)
    return 1-((2* And + smooth) / (keras.sum(y_truef) + keras.sum(y_predf) + smooth))
def dice_coef(y_true,y_pred):
    return 1-dice_loss(y_true,y_pred)

model=Unet()
model.summary()
model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = ['accuracy',dice_coef])
STEP_SIZE_TRAIN=len(train)//Batch_size
STEP_SIZE_VALID=len(test)//Batch_size
callbacks = [ModelCheckpoint('Unet_brain_MRI_seg.hdf5', verbose=1,save_weights_only=True)]
#STEP_SIZE_TEST=len(test_generator)//16
model.load_weights('../input/notebookea74c8185b/Unet_brain_MRI_seg.hdf5')
history=model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=test_gener,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=16
)
model.save_weights('Unet_brain_MRI_seg.hdf5')

key=history.history
list_trainloss=key['loss']
list_testloss=key['val_loss']
plt.figure(1)
plt.plot(list_testloss, 'b-')
plt.plot(list_trainloss,'r-')
plt.legend(["test loss", "train loss"], loc ="upper right")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('loss graph', fontsize = 20)

model.load_weights('Unet_brain_MRI_seg.hdf5')

test_gen = train_generator(df_test, 16,
                               dict(),
                              target_size=(256, 256))
results = model.evaluate(test_gen, steps=len(df_test) / 16)
print(results[0])


for i in range(5):
    index=np.random.randint(1,len(df_test.index))
    img = cv2.imread(df_test['image_path'].iloc[index])
    img = cv2.resize(img ,(256, 256))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(df_test['mask_path'].iloc[index])))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    print(np.max(pred))
    print(np.min(pred))
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    plt.show()




