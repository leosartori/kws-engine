
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from absl import app
from support_functions import *
from IPython.display import Image
from timeit import default_timer as timer
from tensorflow.keras.models import load_model


# LEO: codice per disattivare la GPU, il mio pc sembra avere un problema con i driver e quindi uso CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# stampa la GPU disponibile (non funziona sul cluster)
# print(tf.config.experimental.list_physical_devices('GPU'))


# ---------------------------- PARAMETRI DI INPUT ----------------------------

# flag per selezionare i parametri opportuni per runnare il codice sul cluster DEI
RUN_ON_CLUSTER = True

# select the model to train
# NETWORK_MODEL_TO_TRAIN = 'debug_classifier'
NETWORK_MODEL_TO_TRAIN = 'autoencoder1'
# NETWORK_MODEL_TO_TRAIN = 'encoder_mlp_classifier1'

MODEL_VERSION_TO_TRAIN = 0.1


# select the model to load if a classifier needs to be trained on top of a pre-trained network model
NETWORK_MODEL_TO_LOAD = 'autoencoder1'

MODEL_VERSION_TO_LOAD = 0.0


if RUN_ON_CLUSTER:
    TRAIN_DIR = '/nfsd/hda/DATASETS/Project_1'
    VALIDATION_FILENAME = '/nfsd/hda/DATASETS/Project_1/validation_list.txt'
    TESTING_FILENAME = '/nfsd/hda/DATASETS/Project_1/testing_list.txt'
    BATCH_SIZE = 1024 # 64 nel paper dell'autoencoder
    VERBOSE_FIT = 1  # 0=silent, 1=progress bar, 2=one line per epoch
else:
    # TRAIN_DIR = 'C:/Users/Leonardo/Documents/Uni/HDA/Project/speech_commands_v0.02'
    # TRAIN_DIR = 'C:/Users/Leonardo/Documents/Uni/HDA/Project/debug_dataset_020620/train'
    TRAIN_DIR = 'C:/Users/admin/Desktop/HDA/final_project/dataset/_'
    VALIDATION_FILENAME = './validation_list.txt'
    TESTING_FILENAME = './testing_list.txt'
    BATCH_SIZE = 64 # 64 nel paper dell'autoencoder
    VERBOSE_FIT = 1  # 0=silent, 1=progress bar, 2=one line per epoch


NUM_FEATURES = 40 # number of features per sample (Mel), 320 nel paper degli autoencoder, io proverei 40 nel nostro caso
NUM_RNN_UNITS = 256 # GRU units in encoder and decoder
NUM_MLP_UNITS = 150

LR = 0.001
LR_DROP_FACTOR = 0.5
DROP_EVERY = 10
NUM_EPOCH = 20

# parametri per il calcolo dello spettrogramma (Mel features) a partire da file audio
# nel paper degli autoencoder in valori erano WIN_LEN = 0.2 e WIN_STEP = 0.1 però i file duravano 10 secondi, io userei 25/30ms e 10ms come al solito
MAX_TIMESTEPS_SPECTROGRAMS = 98 # 1sample + (1sec - 0.03sec)/0.01sec = 98 samples
WIN_LEN = 0.03
WIN_STEP = 0.01



# ----------------------------  MAIN --------------------------

def main(argv):

    print()

    # LETTURA DEI FILENAME E CREAZIONE DELLE LABEL
    print('Reading from the dataset folder...')
    print()

    filenames = []
    labels = []
    labels_counter = 0
    labels_dict = {}

    entry_list = os.listdir(TRAIN_DIR)
    entry_list.sort() # ordino perchè os.listdir() restituisce in ordine arbitrario in teoria

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

    X_train_filenames, Y_train, X_val_filenames, Y_val, X_test_filenames, Y_test =\
        split_dataset_from_list(filenames, labels, VALIDATION_FILENAME, TESTING_FILENAME)
    # X_train_filenames, X_val_filenames, Y_train, Y_val = train_test_split(
        # filenames_shuffled, labels_one_hot_shuffled, test_size=0.05, random_state=1)


    # trasformazione delle liste con i filenames in numpy array
    X_train_filenames = np.array(X_train_filenames)
    X_val_filenames = np.array(X_val_filenames)
    X_test_filenames = np.array(X_test_filenames)

    # trasformazione delle liste delle labels in numpy array
    Y_train = np.array(Y_train, dtype=int)
    Y_val = np.array(Y_val, dtype=int)
    Y_test = np.array(Y_test, dtype=int)

    # trasformazione delle label in one hot encoding
    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_val = tf.keras.utils.to_categorical(Y_val)
    Y_test = tf.keras.utils.to_categorical(Y_test)

    NUM_CLASSES = Y_train.shape[1]

    print('Total number of audio files in the dataset: ' + str(len(filenames)))
    print('Total number of classes in the dataset: ' + str(NUM_CLASSES))
    print('Classes: ' + str(labels_dict.values()))
    print('Total number of audio files in the training set: ' + str(X_train_filenames.shape[0]))
    print('Total number of audio files in the validation set: ' + str(X_val_filenames.shape[0]))
    print('Total number of audio files in the test set: ' + str(X_test_filenames.shape[0]))
    print()
    print('Done')
    print()

    # # per selezionare meno file e fare qualche prova di training in locale
    # n_train = 500
    # n_val = 250
    # n_test = 200
    # X_train_filenames = X_train_filenames[0:n_train]
    # X_val_filenames = X_val_filenames[0:n_val]
    # X_test_filenames = X_test_filenames[0:n_test]
    # Y_train = Y_train[0:n_train]
    # Y_val = Y_val[0:n_val]
    # Y_test = Y_test[0:n_test]


    # steps per epoca in modo da analizzare tutto il dataset
    train_steps = int(np.ceil(X_train_filenames.shape[0] / BATCH_SIZE))
    val_steps = int(np.ceil(X_val_filenames.shape[0] / BATCH_SIZE))


    # Network model training



    # if NETWORK_MODEL_TO_TRAIN == 'debug_classifier':
    #
    #     # crea dataset con classe Dataset di TF
    #     train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False,
    #                                    (MAX_TIMESTEPS, NUM_FEATURES), False, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
    #     val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False,
    #                                  (MAX_TIMESTEPS, NUM_FEATURES), False, MAX_TIMESTEPS, WIN_LEN, WIN_STEP)
    #     print('DONE')
    #
    #     # crea e traina il modello con API Keras
    #     print('Creating model...')
    #     debug_classifier = debug_classifier_model(NUM_FEATURES, NUM_UNITS, num_labels)
    #     opt = tf.keras.optimizers.Adam(learning_rate=LR)
    #     debug_classifier.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    #     print('DONE')
    #
    #     print('Training the model:')
    #     debug_classifier.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps,
    #                          validation_data=val_dataset, validation_steps=val_steps, verbose=VERBOSE_FIT)
    #     print('DONE')


    if NETWORK_MODEL_TO_TRAIN == 'autoencoder1':

        # CREAZIONE E TRAIN DEL MODELLO
        print('Creating TF dataset...')
        # per il momento lascio lo shuffle a False perchè ho notato che ci mette un sacco di tempo per farlo (visto in locale, magari nel cluster non è così)
        # mettendo a True con pochi dati ovviamente va veloce lo shuffle, da provare nel cluster

        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, shuffle=True,
                                       input_size=(MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES),
                                       network_model=NETWORK_MODEL_TO_TRAIN,
                                       win_len=WIN_LEN, win_step=WIN_STEP,
                                       normalize=True, cache_file='train_cache')

        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, shuffle=True,
                                     input_size=(MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES),
                                     network_model=NETWORK_MODEL_TO_TRAIN,
                                     win_len=WIN_LEN, win_step=WIN_STEP,
                                     normalize=True, cache_file='val_cache')
        print('Done')
        print()

        # crea e traina il modello con API Keras
        print('Creating the model...')
        rnn_autoencoder = rnn_autoencoder_model(NUM_FEATURES, NUM_RNN_UNITS)


        schedule = StepDecay(init_alpha=LR, factor=LR_DROP_FACTOR, drop_every=DROP_EVERY)
        callbacks = [LearningRateScheduler(schedule, verbose=1)]

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        rnn_autoencoder.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"]) # nel paper viene scritto che usano rmse...
        print('Done')
        print()


        print('Training the model:')
        start_time = timer()

        history = rnn_autoencoder.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps,
                    validation_data=val_dataset, validation_steps=val_steps, callbacks=callbacks, verbose=VERBOSE_FIT)

        end_time = timer()
        load_time = end_time - start_time

        print('Done')
        print()
        printInfo(NETWORK_MODEL_TO_TRAIN, MODEL_VERSION_TO_TRAIN, NUM_FEATURES, BATCH_SIZE, MAX_TIMESTEPS_SPECTROGRAMS,
                  WIN_LEN, WIN_STEP, NUM_EPOCH)
        print('===== TOTAL TRAINING TIME: {0:.1f} sec ====='.format(load_time))
        print()


        # save a plot of the loss/mse trend during the training phase
        save_training_loss_trend_plot(history, NETWORK_MODEL_TO_TRAIN, MODEL_VERSION_TO_TRAIN)

        rnn_autoencoder.save('./training_output/models/' + NETWORK_MODEL_TO_TRAIN + '_v' + str(MODEL_VERSION_TO_TRAIN) + '_.h5')
        print('Model saved to disk')
        print()

        rnn_autoencoder.summary()

        # Showing and saving a picture of the model used
        tf.keras.utils.plot_model(rnn_autoencoder, to_file='./training_output/images/model-plot_' + NETWORK_MODEL_TO_TRAIN + '_v' + str(MODEL_VERSION_TO_TRAIN) + '.png')

        print()
        print("Descrizione del modello per il transfer learning dopo il training dell'autoencoder")
        print(rnn_autoencoder.layers)




    if NETWORK_MODEL_TO_TRAIN == 'encoder_mlp_classifier1':

        # CREAZIONE E TRAIN DEL MODELLO
        print('Creating TF dataset...')
        # per il momento lascio lo shuffle a False perchè ho notato che ci mette un sacco di tempo per farlo (visto in locale, magari nel cluster non è così)
        # mettendo a True con pochi dati ovviamente va veloce lo shuffle, da provare nel cluster

        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, shuffle=True,
                                       input_size=(MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES),
                                       network_model=NETWORK_MODEL_TO_TRAIN,
                                       win_len=WIN_LEN, win_step=WIN_STEP,
                                       normalize=True, cache_file='train_cache')

        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, shuffle=True,
                                     input_size=(MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES),
                                     network_model=NETWORK_MODEL_TO_TRAIN,
                                     win_len=WIN_LEN, win_step=WIN_STEP,
                                     normalize=True, cache_file='val_cache')
        print('Done')
        print()

        # crea e traina il modello con API Keras
        print('Loading the trained autoencoder model...')
        rnn_autoencoder = load_model('./training_output/models/' + NETWORK_MODEL_TO_LOAD + '_v' + str(MODEL_VERSION_TO_LOAD) + '_.h5')
        print('Done')
        print()

        print('Creating the model...')
        encoder_mlp, encoder = rnn_encoder_mlp_model(rnn_autoencoder, NUM_MLP_UNITS, NUM_CLASSES, NUM_FEATURES)

        schedule = StepDecay(init_alpha=LR, factor=LR_DROP_FACTOR, drop_every=DROP_EVERY)
        callbacks = [LearningRateScheduler(schedule, verbose=1)]

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        encoder_mlp.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=["accuracy"])
        print('Done')
        print()

        print('Training the model:')
        start_time = timer()

        # history = encoder_mlp_classifier.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps,
        #                               validation_data=val_dataset, validation_steps=val_steps, callbacks=callbacks,
        #                               verbose=VERBOSE_FIT)

        end_time = timer()
        load_time = end_time - start_time

        print('Done')
        print()
        printInfo(NETWORK_MODEL_TO_TRAIN, MODEL_VERSION_TO_TRAIN, NUM_FEATURES, BATCH_SIZE, MAX_TIMESTEPS_SPECTROGRAMS,
                  WIN_LEN, WIN_STEP, NUM_EPOCH)
        print('===== TOTAL TRAINING TIME: {0:.1f} sec ====='.format(load_time))
        print()


        # # save a plot of the loss/mse trend during the training phase
        # save_training_loss_trend_plot(history, NETWORK_MODEL_TO_TRAIN, MODEL_VERSION_TO_TRAIN)
        # save_training_accuracy_trend_plot(history, NETWORK_MODEL_TO_TRAIN, MODEL_VERSION_TO_TRAIN)
        #
        # encoder_mlp_classifier.save('./training_output/models/' + NETWORK_MODEL_TO_TRAIN + '_v' + str(MODEL_VERSION_TO_TRAIN) + '_.h5')
        # print('Model saved to disk')
        # print()

        print()
        encoder.summary()
        print()
        encoder_mlp.summary()






if __name__ == '__main__':
    app.run(main)