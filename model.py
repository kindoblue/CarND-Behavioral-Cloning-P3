
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalise_dataframe(_df, _nb_bins=10, _sample_per_bin=2000):
    """Normalise the dataframe in respect of steering angles"""

    # get min and max angles in the dataframe
    max_angle = _df['angle'].max()
    min_angle = _df['angle'].min()

    # create equally spaced bins from min angle to max angle
    lin = np.linspace(min_angle, max_angle, _nb_bins)

    # generator for tuples (start, stop) in the entire range
    tuples = ((a, b) for a, b in zip(lin[::2], lin[1::2]))

    # prepare the result dataframe as an empy dataframe with same columns
    res_df = pd.DataFrame(columns=_df.columns)

    # cycle on each slot
    for start, end in tuples:

        # filter dataframe for angles in the brackets
        filtered = _df[(_df['angle'] >= start) & (_df['angle'] <= end)]

        # flag if we need to sample with replacement
        replace = len(filtered) < _sample_per_bin

        # sample the filtered dataframe and append to the output dataframe
        res_df = res_df.append(filtered.sample(_sample_per_bin, replace=replace))

    # we are done
    return res_df


class CNNModel:

    def __init__(self, csv_path):

        # the shape of the images from the simulator
        self.image_shape = (160, 320, 3)

        # column names
        self.columns = ('center', 'left', 'right', 'angle', 'throttle', 'break', 'speed')

        self.angle_corr = {'center': 0.0, 'left': 0.20, 'right': -0.20}

        # path to csv file
        self.csv_path = csv_path

        # save the current path, to be used later when loading
        # the images
        self.current_path = os.path.dirname(csv_path) + '/IMG/'

        self.learning_rate = 2e-4

        self.batch_size = 128

        self.epochs = 8

        self.augmentation_factor = 4

    def load_csv(self):
        """Read the csv and return the train and validation 
           sets as pandas dataframes"""

        # read the dataframe from the cvs
        df = pd.read_csv(self.csv_path, header=None, names=self.columns)

        # normalize the dataframe, boosting the less represented angles
        # at the same levels of others
        norm = normalise_dataframe(df, _nb_bins=10, _sample_per_bin=3000)

        # norm.hist(column='angle', bins=10)

        # return two dataframes, one for training and the other for
        # validation using sklearn
        return train_test_split(norm, test_size=.2)


    def build_model(self):
        """Build the cnn model. Using keras"""

        # create the sequential model
        _model = Sequential()

        # image pre-processing: normalize the image
        _model.add(Lambda(lambda img: img / 255.0 - 0.5,
                          input_shape=self.image_shape))

        # image pre-processing: crop the images to simplify the scene and
        # keep only the relevant parts
        _model.add(Cropping2D(cropping=((55, 0), (0, 0)),
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

        _model.add(Dense(1, name='dense_4'))  # NO RELU AFTER! damn cut&paste

        adam = Adam(lr=self.learning_rate)

        _model.compile(loss='mse', optimizer=adam)

        return _model

    def generator(self, _data, _batch_size, _is_training=False):
        """Returns a generator. For every batch, we choose randomly the
           center, right or left camera, applying the angle correction if
           needed. You get 60% of the times the center camera, remaining 20% 
           right and 20% left"""

        if _is_training:
            # in case of training, we are using an util generator
            # for data augmenting
            datagen = ImageDataGenerator(
                rotation_range=4,
                width_shift_range=0.03,
                height_shift_range=0.03,
                horizontal_flip=False)

        while 1:

            # split in chucks
            for offset in range(0, len(_data), _batch_size):

                # get the nth chunk (slicing dataframe slices it by rows, good)
                sample = _data[offset : offset + _batch_size]

                # select center, right or left columns according a set of probs
                # for experimentation
                col = np.random.choice(self.columns,
                                       1,
                                       p=[1.0, 0, 0, 0, 0, 0, 0])[0]

                # get image paths as an array
                paths = sample[:][col].values

                # adjust the path of the images using the current path stored
                # before (which is the dir containing the csv)
                # so we can run on different systems
                paths = map(lambda x: self.current_path + x.split('/')[-1], paths)

                # preallocate the image array
                shape = (len(sample),) + self.image_shape
                images = np.empty(shape)

                # load the images
                for idx, image_path in enumerate(paths):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    assert(image is not None)  # otherwise we keep going even...
                    images[idx] = image

                # now get the steering angle correction based on the
                # camera we selected
                corr = self.angle_corr[col]

                # adjust the angles and return as a numpy array
                angles = sample[:]['angle'].apply(lambda x: x + corr).values

                if _is_training:
                    for _ in range(self.augmentation_factor):
                        yield datagen.flow(images, angles).next()
                else:
                    yield shuffle(images, angles)

    def train(self):

        # get the train and validation sets
        df_train, df_valid = self.load_csv()

        # create the cnn
        model = self.build_model()

        # print the architecture
        print(model.summary())

        # create the generators that will feed the keras training
        train_gen = self.generator(df_train, _batch_size=self.batch_size, _is_training=True)
        valid_gen = self.generator(df_valid, _batch_size=self.batch_size)

        # verify the crop function
        # crop = K.function([model.layers[0].input], [model.layers[1].output])
        # for a in train_gen:
        #    images = a[0]
        #    cropped_images = crop([images])
             # cropped_images[0] is (32, 105, 320, 3)
        #    plt.imshow(cropped_images[0][0])

        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=3, verbose=1,
                                       mode='auto')
        # train the model
        history = model.fit_generator(train_gen,
                            samples_per_epoch=self.augmentation_factor * len(df_train),
                            validation_data=valid_gen,
                            nb_val_samples=len(df_valid),
                            nb_epoch=self.epochs,
                            callbacks=[checkpoint, early_stopping])

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    # the_model = CNNModel('/Users/ice/Development/AA/run1/driving_log.csv')
    the_model = CNNModel('/home/carnd/run1/driving_log.csv')

    the_model.train()
