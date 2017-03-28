from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D

class CNNModel:

    def __init__(self):
        self.image_shape = (160, 320, 3)

    def build_model(self):
        """Build the cnn model. Using keras"""

        # create the sequential model
        _model = Sequential()

        # image preprocessing: normalize the image
        _model.add(Lambda(lambda img: img / 255.0 - 0.5,
                          input_shape=self.image_shape))

        # image preprocessing: crop the images to simplify the scene and
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

        _model.add(Dense(1,
                         activation='relu',
                         name='dense_4'))

        _model.compile(loss='mse', optimizer='adam')

        return _model




    def train(self):

        # create the cnn
        model = self.build_model()


if __name__ == '__main__':
    the_model = CNNModel()
    the_model.train()
