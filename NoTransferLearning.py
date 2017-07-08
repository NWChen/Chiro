import keras
import os
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

numClasses = 2
imgChannels = 3
imgCols, imgRows = 128, 128
batchSize = 64
numEpochs = 15

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
        fileNames = os.listdir(str(positiveTrainFolder + folder))
        positiveFileNames += [folder + '/' + f for f in fileNames]
        break
    for folder in negativeSubFolders:
        fileNames += os.listdir(negativeTrainFolder + folder)
        negativeFileNames += [folder + '/' + f for f in fileNames]
        break

    numTrainExamples = len(positiveFileNames) + len(negativeFileNames)
    X = np.ones((numTrainExamples,imgCols,imgRows,imgChannels))
    Y = np.ones((numTrainExamples, numClasses))

    for index,filename in enumerate(positiveFileNames):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            imgPath = str(positiveTrainFolder + filename) 
            img = image.load_img(imgPath, target_size=(imgCols, imgRows))
            img = image.img_to_array(img)
            X[index] = img
            Y[index] = [1,0]
        else:
            print('Something went wrong')

    for index,filename in enumerate(negativeFileNames):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            imgPath = str(negativeTrainFolder + filename) 
            img = image.load_img(imgPath, target_size=(imgCols, imgRows))
            img = image.img_to_array(img)
            X[index] = img
            Y[index] = [0,1]
        else:
            print('Something went wrong')
    return X,Y

X,Y = processDataset()
print X.shape
print Y.shape 
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=xTrain.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

datagen = ImageDataGenerator(horizontal_flip=True, zca_whitening=True, width_shift_range=0.2,
        height_shift_range=0.2, zoom_range=0.2) 
datagen.fit(xTrain)

model.fit_generator(datagen.flow(xTrain, yTrain), samples_per_epoch=len(xTrain), epochs=numEpochs, validation_data=(xTest, yTest))
score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.h5')

