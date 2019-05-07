from __future__ import print_function
from __future__ import absolute_import

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.layers import Input, np
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold


class Model_resNet():
    def __init__(self, args, load=False):
        self.ckpt = args.pre_train
        self.model = "ResNet"
        self.args = args
        self.class_num = args.class_num
        self.lr = args.lr
        self.epoch = args.epoch
        self.c = args.n_color
        self.is_online = args.online
        self.batch_size = args.batch_size
        self.save_dir = args.save
        self.sess = None
        self.is_test = load
        # self.mode = args.processing_mode

        self.callbacks = []

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def DeepModel(self, size_set=224):
        img_input = Input(shape=(size_set, size_set, 3))
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten(name='out_feat')(x)
        x = Dense(3, activation='softmax', name='fc2')(x)

        # Create model.
        model = Model(img_input, x)

        return model

    def init_callbacks(self, name):
        self.callbacks = []
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.save_dir + "ResNet" + '_best_weights' + str(name) + '.h5',
                verbose=1,
                monitor='val_categorical_accuracy',
                mode='auto',
                save_best_only=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.args.save,
                write_images=True,
                write_graph=True,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                # patience=self.args.early_stopping
                patience=self.args.early_stopping
            )
        )

        # self.callbacks.append(
        #    ReduceLROnPlateau(
        #        monitor='val_loss',
        #        factor=0.5,
        #        patience=5,
        #        min_lr=1e-5
        #    )
        # )

        def custom_schedule(epochs):
            if epochs <= 5:
                lr = 1e-3
            elif epochs <= 50:
                lr = 5e-4
            elif epochs <= 100:
                lr = 2.5e-4
            elif epochs <= 500:
                lr = 1e-4
            elif epochs <= 700:
                lr = 5e-5
            else:
                lr = 1e-5

            return lr

        self.callbacks.append(
            LearningRateScheduler(
                custom_schedule
            )
        )

    def train(self, training_images, training_labels, validation_images, validation_labels):
        self.init_callbacks('')
        self.model = self.DeepModel()
        self.model.trainable = True
        self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
                           loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        if len(validation_images) == 0:
            print('no val')
            validation_images = training_images[350:]
            validation_labels = training_labels[350:]
            training_labels = training_labels[:350]
            training_images = training_images[:350]

        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        val_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        steps = int(np.size(training_images, 0) // self.batch_size)
        val_steps = int(np.size(validation_images, 0) // self.batch_size)

        self.model.fit_generator(
            generator=train_datagen.flow(x=training_images, y=training_labels, batch_size=self.batch_size),
            epochs=self.args.epoch,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=val_datagen.flow(x=validation_images, y=validation_labels, batch_size=self.batch_size)
        )

    def load_data_kfold(self, k, training_images, training_labels):
        X_train = training_images
        y_train = training_labels
        folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))
        return folds, X_train, y_train

    def train_5fold(self, training_images, training_labels):
        self.model = self.DeepModel()
        self.model.trainable = True
        self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
                           loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # self.model.load_weights("D:\Projects\jiaomo-master\Model\model5_resNet5fold\ResNet_best_weights_fold_0.h5")
        k = 5
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        val_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        folds, x_train, y_train = self.load_data_kfold(k, training_images, training_labels)
        # print(folds)
        print(len(training_images))
        print(len(x_train))
        for j, (train_idx, val_idx) in enumerate(folds):
            print('\nFold ', j)
            x_train_cv = x_train[train_idx]
            y_train_cv = y_train[train_idx]
            y_train_cv = to_categorical(np.array(y_train_cv))
            print(len(x_train_cv))

            x_valid_cv = x_train[val_idx]
            y_valid_cv = y_train[val_idx]
            print(len(x_valid_cv))

            y_valid_cv = to_categorical(np.array(y_valid_cv))
            steps = int(np.size(x_train_cv, 0) // self.batch_size)
            val_steps = int(np.size(x_valid_cv, 0) // self.batch_size)
            name_weights = "_fold_" + str(j)
            self.init_callbacks(name=str(name_weights))
            self.model.fit_generator(
                generator=train_datagen.flow(x=x_train_cv, y=y_train_cv, batch_size=self.batch_size),
                epochs=self.args.epoch,
                steps_per_epoch=steps,
                validation_steps=val_steps,
                verbose=1,
                callbacks=self.callbacks,
                validation_data=val_datagen.flow(x=x_valid_cv, y=y_valid_cv, batch_size=self.batch_size)
            )
            print(self.model.evaluate(x_valid_cv, y_valid_cv))
