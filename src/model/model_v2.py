from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, \
    LearningRateScheduler
import numpy as np
from keras.models import model_from_json


def add_new_last_layer(base_model, nb_classes, fc_size=1024):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    # 如果你使用的是 0-1非One Hot编码 使用下句
    # predictions = Dense(1, activation='sigmoid')(x)  # new sigmoid layer
    model = Model(input=base_model.input, output=predictions)
    return model


class Model_v2():
    def __init__(self, args, load=False):
        self.ckpt = args.pre_train
        self.data_dir = args.data_path + '/hdf5'
        self.dataset = args.dataset
        self.model = args.model
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
        self.mode = args.processing_mode
        self.callbacks = []
        self.init_callbacks()
        self.model = None
        if args.model == 'inception':
            self.W = 299
            self.H = 299
        elif args.model == 'DR':
            self.W = 331
            self.H = 331
        if not self.is_test:
            self.fine_tune()
        else:
            self.load()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.args.save + self.args.model + '_best_weights.h5',
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
            elif epochs <= 10:
                lr = 5e-4
            elif epochs <= 20:
                lr = 2.5e-4
            else:
                lr = 1e-4

            return lr

        self.callbacks.append(
            LearningRateScheduler(
                custom_schedule
            )
        )

    def load(self):
        self.model = model_from_json(
            open(self.args.save + self.args.model + '_architecture.json').read())

    def fine_tune(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        self.model = add_new_last_layer(base_model, self.class_num)
        self.model.trainable = True
        self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.1),
                           loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def train(self, training_images, training_labels, validation_images, validation_labels):

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
            validation_data=val_datagen.flow(x=validation_images, y=validation_labels, batch_size=self.batch_size))
