import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np
from keras.optimizers import SGD
import sys

# Variables and hyperparameters
numClasses = 2
imgChannels = 3
imgCols, imgRows = 299, 299
batchSize = 64
layersToFreeze = 172
batchSize = 32
numInitialEpochs = 16
numFinalEpochs = 3

# Goes through the pictures in the folder, and creates training dataset
def processDataset():
    positiveTrainFolder = 'data/good/'
    negativeTrainFolder = 'data/bad/'
    positiveSubFolders = os.listdir(positiveTrainFolder)
    negativeSubFolders = os.listdir(negativeTrainFolder)
    # Some weird file that gets added in with Unix to Mac transfer
    if ('.DS_Store' in positiveSubFolders):
        positiveSubFolders.remove('.DS_Store')
    if ('.DS_Store' in negativeSubFolders):
        negativeSubFolders.remove('.DS_Store')
    
    positiveFileNames=[]
    negativeFileNames=[]
    for folder in positiveSubFolders:
        fileNames = os.listdir(positiveTrainFolder + folder)
        positiveFileNames += [folder + '/' + f for f in fileNames]
    for folder in negativeSubFolders:
        fileNames += os.listdir(negativeTrainFolder + folder)
        negativeFileNames += [folder + '/' + f for f in fileNames]

    numTrainExamples = len(positiveFileNames) + len(negativeFileNames)
    X = np.ones((numTrainExamples,imgCols,imgRows,imgChannels))
    Y = np.ones((numTrainExamples, numClasses))

    for index,filename in enumerate(positiveFileNames):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            imgPath = str(positiveTrainFolder + filename) # TODO double check if this is right
            img = image.load_img(imgPath, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            X[index] = img
            Y[index] = [1,0]
        else:
            print('Something went wrong')

    for index,filename in enumerate(negativeFileNames):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            imgPath = str(negativeTrainFolder + filename) # TODO double check if this is right
            img = image.load_img(imgPath, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            X[index] = img
            Y[index] = [0,1]
        else:
            print('Something went wrong')
    return X,Y

# Wanted to check if this is actually working the way we want it to
def testImageDataGenerator():
 	datagen = ImageDataGenerator(horizontal_flip=True, zca_whitening=True, width_shift_range=0.2,
    	height_shift_range=0.2, zoom_range=0.2) 
	datagen.fit(xTrain)
	i = 0
	for batch in datagen.flow(xTrain, batch_size=1, save_to_dir='preview', save_format='jpeg'):
		i += 1
		print i
    	if i > 20:
    		sys.exit(0)  # otherwise the generator would loop indefinitely

# Create the training dataset
X,Y = processDataset()
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.20)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer because apparently that's helpful??
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer
predictions = Dense(numClasses, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
	print layer
   	layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(xTrain, yTrain, batch_size=batchSize, nb_epoch=numInitialEpochs,validation_data=(xTest, yTest),shuffle=True)

# At this point, the top layers are well trained and we can start fine-tuning

for layer in model.layers[:layersToFreeze]:
   layer.trainable = False
for layer in model.layers[layersToFreeze:]:
   layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# Keras has a cool object that creates augmented data for you just in time
# during the training process, and thus you don't have to save it in memory.

#datagen = ImageDataGenerator(horizontal_flip=True, zca_whitening=True, width_shift_range=0.2,
    	#height_shift_range=0.2, zoom_range=0.2, ) 
#datagen.fit(xTrain)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(datagen, samples_per_epoch=len(xTrain), epochs=numFinalEpochs, validation_data=(xTest, yTest))
model.fit(xTrain, yTrain, batch_size=batchSize, nb_epoch=numFinalEpochs,shuffle=True)
model.evaluate(xTest, yTest, batch_size=batchSize)

# Saving the model

# Loading the model 