from __future__ import absolute_import
from __future__ import print_function

from keras.applications.nasnet import NASNetMobile
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.layers import np
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator


class Model_Nas:
    def __init__(self, args, load=False):
        self.ckpt = args.pre_train
        self.model = "NasNet"
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
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.save_dir + self.model + '_best_weights.h5',
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
                patience=1000
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
        self.model = NASNetMobile(classes=2, include_top=True, weights=None)
        self.model.trainable = True
        self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
                           loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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
