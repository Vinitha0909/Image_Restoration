# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
train_fpath = "../input/denoising/denoising-dirty-documents/train/train/"
train_cleaned_fpath = "../input/denoising/denoising-dirty-documents/train_cleaned/train_cleaned/"
test_fpath = "../input/denoising/denoising-dirty-documents/test/test/"
print(os.listdir(train_fpath))
print("No. of files in train folder = ",len(os.listdir(train_fpath)))
print("No. of files in train_cleaned folder = ",len(os.listdir(train_cleaned_fpath)))
print("No. of files in test folder = ",len(os.listdir(test_fpath)))
def load_images(fpath):
    images = []
    for image in os.listdir(fpath):
        if image!='train' and image!='train_cleaned' and image!='test':
            img = cv2.imread(fpath+image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(img, "RGB")
            resized_img = img_array.resize((252,252))
            images.append(np.array(resized_img))
    return images
#print(len(images))
train_images = load_images(train_fpath)
train_images = np.array(train_images)
print((train_images.size()))
print("No. of images loaded = ",len(train_images),"Shape of the images loaded =",train_images[0].shape)
train_cleaned_images = load_images(train_cleaned_fpath)
train_cleaned_images = np.array(train_cleaned_images)
print("No. of images loaded = ",len(train_cleaned_images),"Shape of the imagesloaded = ",train_cleaned_images[0].shape)
test_images = load_images(test_fpath)
test_images = np.array(test_images)
print("No. of images loaded = ",len(test_images),"Shape of the images loaded =",test_images[0].shape)

def display_images(images):
    n = 5
    plt.figure(figsize=(19, 6))
    for i in range(n):
        ax = plt.subplot(1,n , i+1)
        plt.imshow(images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
print("Displaying noisy training images")
display_images(train_images)

#Data normalization
train_images = train_images.astype(np.float32)
train_cleaned_images = train_cleaned_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_images = train_images/255
train_cleaned_images = train_cleaned_images/255
test_images = test_images/255
print(train_images[0].shape, train_cleaned_images[0].shape, test_images[0].shape)

print("Displaying noisy training images after normalization")
display_images(train_images)

print("Displaying clean training images after normalization")
display_images(train_cleaned_images)

#Define auto encoder
input_img = Input(shape=(252, 252, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()
autoencoder.fit(train_images, train_cleaned_images,epochs=100,batch_size=100,shuffle=True)

predicted_images = autoencoder.predict(test_images)
#Display noisy test images
print("Displaying noisy test images")
display_images(test_images)
#Display clean images predicted by the autoencoder for the given test images input
print("Displaying predicted images for the given test noisy images input")
display_images(predicted_images)



