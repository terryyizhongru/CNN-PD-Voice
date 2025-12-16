#!/usr/bin/env python
# coding: utf-8

# Inception Model for Parkinson's Disease detection from voice spectral data.  
# The Inception V3 model, pretrained on Imagenet is adapted using transfer learning to
# extract features from spectrogram images of the sustained vowel /a/ to distinguish people 
# with Parkinson’s Disease (PwPD) from healthy controls (HC).  
# 
# Audio files were preprocessed using the R packages Create_Liner(Mel)Spectrograms_(dataset) which are available in https://github.com/uams-tri/PD-Voice.
# 
# Spectra files must be oranized into the directory structure required by the Keras ImageDatatGenerator() class when using the
# class_mode- 'categorical' option as described in 
# https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720.
# The pathname to this directory strucutre must be changed for each data set being analyzed. Output file names should be modified per experiment.
# This approach was used to improve understandability.
# For comparison with other analyses performeed on these data, training for transfer learing is re-initialized on each of 100 iterations retaining ROC data.


# Copyright (C) 2024 University of Arkansas for Medical Sciences
# Author: Anu Iyer, Fred Prior, PhD FWPrior@uams.edu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


import pandas as pd
import numpy as np
import tensorflow as tf
import timeit
import datetime
import math
import re
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def wav_to_jpg_path(wav_path: str) -> str:
    # User rule: wavpath.replace('fortrain', 'fortrain_jpg').replace('.wav', '.jpg')
    jpg_path = wav_path.replace('fortrain', 'fortrain_jpg')
    jpg_path = re.sub(r"\.wav$", ".jpg", jpg_path, flags=re.IGNORECASE)
    return jpg_path


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"healthy", "hc", "control", "controls"}:
        return "healthy"
    if s in {"parkinson", "pd", "parkinsons", "parkinson's"}:
        return "parkinson"
    return s


def load_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t')
    df = df.copy()
    df["filename"] = df["AUDIOFILE"].astype(str).apply(wav_to_jpg_path)
    df["label"] = df["DIAGNOSIS"].astype(str).apply(normalize_label)
    return df


def build_data_generators(data_path, img_rows, img_cols, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
        validation_split=0.3,
    )
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )
    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return train_generator, validation_generator


def build_data_generators_from_tsv(
    train_tsv_path,
    img_rows,
    img_cols,
    batch_size,
    *,
    val_tsv_path=None,
    validation_split=0.3,
    seed=123,
):
    if val_tsv_path is None:
        df = load_tsv(train_tsv_path)
        classes = sorted(df["label"].unique().tolist())
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            fill_mode='nearest',
            validation_split=validation_split,
        )
        train_generator = datagen.flow_from_dataframe(
            df,
            x_col="filename",
            y_col="label",
            classes=classes,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training',
            seed=seed,
        )
        validation_generator = datagen.flow_from_dataframe(
            df,
            x_col="filename",
            y_col="label",
            classes=classes,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation',
            seed=seed,
        )
        return train_generator, validation_generator

    df_train = load_tsv(train_tsv_path)
    df_val = load_tsv(val_tsv_path)
    classes = sorted(df_train["label"].unique().tolist())

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
    )
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        fill_mode='nearest',
    )
    validation_generator = val_datagen.flow_from_dataframe(
        df_val,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return train_generator, validation_generator


def build_eval_generator_from_tsv(tsv_path, img_rows, img_cols, batch_size, classes):
    df = load_tsv(tsv_path)
    datagen = ImageDataGenerator(rescale=1.0 / 255, fill_mode='nearest')
    return datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="label",
        classes=classes,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )


def build_model():
    pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=(600, 600, 3), pooling='avg')
    for layer in pre_trained.layers:
        layer.trainable = False

    x = pre_trained.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=pre_trained.input, outputs=predictions)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
    )
    return model


def run_single_experiment(train_generator, validation_generator, batch_size, epochs):
    model = build_model()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath='mPowerHiFreqData2model_{accuracy:.3f}.h5',
                save_best_only=True,
                save_weights_only=False,
                monitor='accuracy',
            )
        ],
    )

    y_scores = model.predict(validation_generator, batch_size)
    y_pred = np.argmax(y_scores, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(validation_generator.classes, y_pred)
    auc_value = metrics.auc(fpr, tpr)
    print("auc: {}".format(round(auc_value, 2)))
    print("\n")
    return model, history, auc_value, y_scores


def summarize_results(model, eval_generator, batch_size, all_auc):
    print("\n")
    model.save("mPowerHiFreqData2Best.h5")
    Y_pred = model.predict(eval_generator, batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(eval_generator.classes, y_pred))
    print('Classification Report:')
    class_indices = getattr(eval_generator, "class_indices", {})
    if class_indices:
        target_names = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]
    else:
        target_names = ['healthy', 'parkinson']
    print(classification_report(eval_generator.classes, y_pred, target_names=target_names))
    print("avg auc: {} ({})".format(np.round(np.average(all_auc), 4), np.round(np.std(all_auc), 3)))


def plot_history(history, all_auc):
    accs = history.history['accuracy']
    val_accs = history.history['val_accuracy']
    plt.plot(range(len(accs)), accs, label='Training_accuracy')
    plt.plot(range(len(accs)), val_accs, label='Validation_accuracy')
    plt.legend()
    plt.show()

    losses = history.history['loss']
    val_losses = history.history['val_loss']
    plt.plot(range(len(losses)), losses, label='Training_loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation_loss')
    plt.legend()
    plt.show()

    print('AUC vector: ', all_auc)
    DF = pd.DataFrame(all_auc)
    DF.to_csv("mPowerHiFreqData2AUC.csv")


def main():
    start = timeit.default_timer()

    # Optional TSV inputs (minimal-change support):
    # TSV columns: ID, AUDIOFILE, DIAGNOSIS
    # AUDIOFILE is wav path; jpg path is derived by:
    #   wavpath.replace('fortrain', 'fortrain_jpg').replace('.wav', '.jpg')
    train_tsv_path = "/home/yzhong/data/storage2/gits/CNN-PD-Voice/split_5fold/folds_v2.1_early_validation_newcut/folds_tsv_SUSTAINED-VOWELS_onlyA123/fold_1/test_early6PD6HC.tsv"
    val_tsv_path = train_tsv_path
    test_tsv_path = train_tsv_path

    img_rows = 600
    img_cols = 600
    batch_size = 4
    epochs = 10
    n_runs = 1

    all_auc = []
    last_history = None
    last_model = None
    train_generator = None
    validation_generator = None
    eval_generator = None
    tsv_classes = None

    for run in range(n_runs):
        print("run={}".format(run))
        if train_tsv_path is None:
            train_generator, validation_generator = build_data_generators(
                data_path, img_rows, img_cols, batch_size
            )
        else:
            train_generator, validation_generator = build_data_generators_from_tsv(
                train_tsv_path,
                img_rows,
                img_cols,
                batch_size,
                val_tsv_path=val_tsv_path,
            )
            tsv_classes = [name for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1])]
            if test_tsv_path is not None:
                eval_generator = build_eval_generator_from_tsv(
                    test_tsv_path, img_rows, img_cols, batch_size, classes=tsv_classes
                )
            else:
                eval_generator = validation_generator

        model, history, auc_value, y_scores = run_single_experiment(
            train_generator, validation_generator, batch_size, epochs
        )
        all_auc.append(auc_value)
        last_history = history
        last_model = model

    if last_model is not None and validation_generator is not None:
        summarize_results(last_model, eval_generator or validation_generator, batch_size, all_auc)
        plot_history(last_history, all_auc)

    stop = timeit.default_timer()
    print('RunTime: ', round(stop - start, 2), 'Seconds')


if __name__ == "__main__":
    main()

