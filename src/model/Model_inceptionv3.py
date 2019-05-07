from __future__ import absolute_import
from __future__ import print_function

from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.layers import np, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold


class Model_inception_v3:
    def __init__(self, args, load=False):
        self.ckpt = args.pre_train
        self.model = "inception_v3"
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

    def add_new_last_layer(self, base_model, nb_classes, fc_size=1024):
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

    def init_callbacks(self, name):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.save_dir + "Inception_v3" + '_best_weights' + str(name) + '.h5',
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
                patience=100
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
        self.model = InceptionV3(include_top=False, weights='imagenet')
        self.model = self.add_new_last_layer(self.model, nb_classes=2)
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
        self.model = InceptionV3(include_top=False, weights='imagenet')
        self.model = self.add_new_last_layer(self.model, nb_classes=2)
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
