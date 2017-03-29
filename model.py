from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, Cropping2D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np
import cv2

class CNNModel:

    def __init__(self):
        self.image_shape = (160, 320, 3)
        self.current_path = ''

    def load_csv(self, _path_to_csv):
        """Read the csv and return the train and validation 
           sets as pandas dataframes"""

        # save the current path, to be used later when loading
        # the images
        self.current_path = os.path.dirname(_path_to_csv)

        # read the dataframe from the cvs
        df = pd.read_csv(_path_to_csv,
                         header=None,
                         names=('center', 'left', 'right', 'angle',
                                'throttle', 'break', 'speed'))

        # return two dataframes, one for training and the other for
        # validation using sklearn
        return train_test_split(df, test_size=.2)


    def build_model(self):
        """Build the cnn model. Using keras"""

        # create the sequential model
        _model = Sequential()

        # image pre-processing: normalize the image
        _model.add(Lambda(lambda img: img / 255.0 - 0.5,
                          input_shape=self.image_shape))

        # image pre-processing: crop the images to simplify the scene and
        # keep only the relevant parts
        _model.add(Cropping2D(cropping=((60, 20), (0, 0)),
                              input_shape=self.image_shape))

        # cnn from nvidia paper:
        # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        _model.add(Convolution2D(24, 5, 5,
                                 subsample=(2, 2),
                                 activation='relu',
                                 name='conv1'))

        _model.add(Convolution2D(36, 5, 5,
                                 subsample=(2, 2),
                                 activation='relu',
                                 name='conv2'))

        _model.add(Convolution2D(48, 5, 5,
                                 subsample=(2, 2),
                                 activation='relu',
                                 name='conv3'))

        _model.add(Convolution2D(64, 3, 3,
                                 subsample=(1, 1),
                                 activation='relu',
                                 name='conv4'))

        _model.add(Convolution2D(64, 3, 3,
                                 subsample=(1, 1),
                                 activation='relu',
                                 name='conv5'))

        _model.add(Flatten())

        _model.add(Dense(100,
                         activation='relu',
                         name='dense_1'))

        _model.add(Dense(50,
                         activation='relu',
                         name='dense_2'))

        _model.add(Dense(10,
                         activation='relu',
                         name='dense_3'))

        _model.add(Dense(1, name='dense_4'))

        _model.compile(loss='mse', optimizer='adam')

        return _model

    def generator(self, _data, _batch_size):
        """Returns a generator"""
        while 1:

            # split in chucks
            for offset in range(0, len(_data), _batch_size):

                # get the nth chunk (slicing dataframe slices it by rows, good)
                sample = _data[offset : offset + _batch_size]

                # get image paths as an array
                paths = sample[:]['center'].values

                # adjust the path of the images using the current path stored
                # before (which is the dir containing the csv)
                # so we can run on different systems
                paths = map(lambda x: self.current_path + '/IMG/' + x.split('/')[-1], paths)

                # preallocate the image array
                shape = (len(sample),) + self.image_shape
                images = np.empty(shape)

                # load the images
                for idx, image_path in enumerate(paths):
                    image = cv2.imread(image_path)
                    images[idx] = image

                # now get the steering angles
                angles = sample[:]['angle'].values

                # return the batch shuffled
                yield shuffle(images, angles)


    def train(self):

        # get the train and validation sets
        df_train, df_valid = self.load_csv('/Users/ice/Development/driving/driving_log.csv')

        # create the cnn
        model = self.build_model()

        # create the generators that will feed the keras training
        train_gen = self.generator(df_train, _batch_size=32)
        valid_gen = self.generator(df_valid, _batch_size=32)

        # train the model
        model.fit_generator(train_gen,
                            samples_per_epoch=len(df_train),
                            validation_data=valid_gen,
                            nb_val_samples=len(df_valid),
                            nb_epoch=8)

        # save the model
        model.save('model.h5')


if __name__ == '__main__':
    the_model = CNNModel()
    the_model.train()
