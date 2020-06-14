import os

# LEO: codice per disattivare la GPU, il mio pc sembra avere un problema con i driver e quindi uso CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from support_functions import create_dataset, rnn_model, debug_classifier_model, StepDecay

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from absl import app

# stampa la GPU disponibile (non funziona sul cluster)
# print(tf.config.experimental.list_physical_devices('GPU'))

# ---------------------------- PARAMETRI DI INPUT ----------------------------
# flag per selezionare i parametri opportuni per runnare il codice sul cluster DEI
RUN_ON_CLUSTER = False

# scegliere se usare come modello il classificatore di debug oppure l'autoencoder
DEBUG_CLASSIFIER = False

if RUN_ON_CLUSTER:
    TRAIN_DIR = "/nfsd/hda/DATASETS/Project_1"
    BATCH_SIZE = 1024
    VERBOSE_FIT = 1  # 0=silent, 1=progress bar, 2=one line per epoch
else:
    # TRAIN_DIR = "C:/Users/Leonardo/Documents/Uni/HDA/Project/speech_commands_v0.02"
    TRAIN_DIR = "C:/Users/Leonardo/Documents/Uni/HDA/Project/debug_dataset_020620/train"
    BATCH_SIZE = 8
    VERBOSE_FIT = 1  # 0=silent, 1=progress bar, 2=one line per epoch

NUM_FEATURES = 320  # number of features per sample (Mel)
NUM_UNITS = 256  # GRU units in encoder and decoder
LR = 0.01
NUM_EPOCH = 10
# Epoca 45: loss = 20 stabile (possibile sia qui necessario LR=0.001)

# parametri per il calcolo dello spettrogramma (Mel features) a partire da file audio
MAX_TIMESTEPS = 9
WIN_LEN = 0.2
WIN_STEP = 0.1


# ----------------------------  MAIN --------------------------

def main(argv):

    # LETTURA DEI FILENAME E CREAZIONE DELLE LABEL
    print('Reading dataset...')

    filenames = []
    labels = []
    labels_counter = 0
    labels_dict = {}

    entry_list = os.listdir(TRAIN_DIR)
    entry_list.sort() # faccio così perchè os.listdir() restituisce in ordine arbitrario in teoria

    for entry in entry_list:

        # skipping files in root directory and background noise folder
        # (non dovrebbe essere una classe ma era usata solo per aggiungere rumore mi sembra)
        if (os.path.isfile(TRAIN_DIR + '/' + entry) is True) or (entry == '_background_noise_'):
            continue

        labels_dict[labels_counter] = entry

        for file in os.listdir(TRAIN_DIR + '/' + entry):

            if file.lower().endswith('.wav'):
                filenames.append(TRAIN_DIR + '/' + entry + '/' + file)
                labels.append(labels_counter)

        labels_counter += 1

    # trasformazione delle liste dei filename e labels in numpy array
    filenames_numpy = np.array(filenames)
    labels_numpy = np.array(labels)

    # trasformazione delle label in one hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels_numpy)
    # TODO: questa operazione penso si possa fare al momento della creazione del Dataset, vedi sotto

    # shuffling dei dati
    filenames_shuffled, labels_one_hot_shuffled = shuffle(filenames_numpy, labels_one_hot)

    # TODO: implementa la divisione dei dati utilizzando which_set o i file validation_list o testing_list già disponibili nel cluster
    X_train_filenames, X_val_filenames, Y_train, Y_val = train_test_split(
        filenames_shuffled, labels_one_hot_shuffled, test_size=0.05, random_state=1)

    # conversione dei vettori di label da float32 a int (negli step precendenti avviene la conversione, bisognerebbe scoprire dove)
    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    num_labels = Y_train.shape[1]

    print('DONE')
    print('Total number of audio files in the dataset: ' + str(filenames_numpy.shape[0]))
    print('Total number of classes in the dataset: ' + str(num_labels))
    print('Classes: ' + str(labels_dict.values()))
    print('Total number of audio files in the training set: ' + str(X_train_filenames.shape[0]))
    print('Total number of audio files in the validation set: ' + str(X_val_filenames.shape[0]))

    # CREAZIONE E TRAIN DEL MODELLO
    print('Creating TF dataset...')

    # steps per epoca in modo da passare tutto il dataset
    train_steps = int(np.ceil(X_train_filenames.shape[0] / BATCH_SIZE))
    val_steps = int(np.ceil(X_val_filenames.shape[0] / BATCH_SIZE))

    # traina classificatore
    if DEBUG_CLASSIFIER:

        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False,
                                       (MAX_TIMESTEPS, NUM_FEATURES), False, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False,
                                     (MAX_TIMESTEPS, NUM_FEATURES), False, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
        print('DONE')

        # crea e traina il modello con API Keras
        print('Creating model...')
        debug_classifier = debug_classifier_model(NUM_FEATURES, NUM_UNITS, num_labels)
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        debug_classifier.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
        print('DONE')

        print('Training the model:')
        debug_classifier.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps,
                             validation_data=val_dataset, validation_steps=val_steps, verbose=VERBOSE_FIT)
        print('DONE')

    # traina autoencoder
    else:
        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False,
                                       (MAX_TIMESTEPS, NUM_FEATURES), True, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False,
                                     (MAX_TIMESTEPS, NUM_FEATURES), True, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
        print('DONE')

        # crea e traina il modello con API Keras
        print('Creating model...')
        autoenc, _ = rnn_model(NUM_FEATURES, NUM_UNITS)

        schedule = StepDecay(init_alpha=0.01, factor=0.9, drop_every=1)
        callbacks = [LearningRateScheduler(schedule)]

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        autoenc.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])
        print('DONE')

        print('Training the model:')
        autoenc.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=callbacks, verbose=VERBOSE_FIT)
        print('DONE')


if __name__ == '__main__':
    app.run(main)
