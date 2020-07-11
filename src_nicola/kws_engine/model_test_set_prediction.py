
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
RUN_ON_CLUSTER = False

# select the model to load if a classifier needs to be trained on top of a pre-trained network model
NETWORK_MODEL_TO_LOAD = 'encoder_mlp_classifier1'

MODEL_VERSION_TO_LOAD = 0.1


if RUN_ON_CLUSTER:
    TRAIN_DIR = '/nfsd/hda/DATASETS/Project_1'
    VALIDATION_FILENAME = '/nfsd/hda/DATASETS/Project_1/validation_list.txt'
    TESTING_FILENAME = '/nfsd/hda/DATASETS/Project_1/testing_list.txt'
    BATCH_SIZE = 64 # 64 nel paper dell'autoencoder
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

# parametri per il calcolo dello spettrogramma (Mel features) a partire da file audio
# nel paper degli autoencoder in valori erano WIN_LEN = 0.2 e WIN_STEP = 0.1 però i file duravano 10 secondi, io userei 25/30ms e 10ms come al solito
MAX_TIMESTEPS_SPECTROGRAMS = 98 # 1sample + (1sec - 0.03sec)/0.01sec = 98 samples
WIN_LEN = 0.03
WIN_STEP = 0.01


FEATURES_TYPES= ['cepstral', 'mel-spectrogram', 'log-mel-spectrogram', 'mel-spectrogram-Audeep']
FEATURES_CHOICE = 2
# each features will be automatically normalized between -1 and 1 in the function compute_spectrogram()
# 'mel-spectrogram-Audeep' cannot be selected because I don't undestand why the time length of the output is half


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

    _, _, _, _, X_test_filenames, Y_test =\
        split_dataset_from_list(filenames, labels, VALIDATION_FILENAME, TESTING_FILENAME)
    # X_train_filenames, X_val_filenames, Y_train, Y_val = train_test_split(
        # filenames_shuffled, labels_one_hot_shuffled, test_size=0.05, random_state=1)

    # trasformazione delle liste con i filenames in numpy array
    X_test_filenames = np.array(X_test_filenames)

    # trasformazione delle liste delle labels in numpy array
    Y_test = np.array(Y_test, dtype=int)


    # crea dataset con classe Dataset di TF
    test_dataset = create_dataset(X_test_filenames, Y_test, NUM_FEATURES, BATCH_SIZE,
                                   input_size=(MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES),
                                   network_model=NETWORK_MODEL_TO_LOAD,
                                   win_len=WIN_LEN, win_step=WIN_STEP, feature_type=FEATURES_TYPES[FEATURES_CHOICE],
                                   shuffle=False,
                                   tensor_normalization=False, cache_file='train_cache', mode='test')


    print('Done')
    print()

    print('Loading the trained classification model...')
    classification_model = load_model(
        './training_output/models/' + NETWORK_MODEL_TO_LOAD + '_v' + str(MODEL_VERSION_TO_LOAD) + '.h5')

    print()
    classification_model.summary()
    print()
    print('Done')
    print()

    print('Making prediction on the test set:')
    start_time = timer()

    test_loss, test_acc = classification_model.evaluate(test_dataset)

    print()
    print('Test loss: ' + str(test_loss))
    print('Test accuracy: ' + str(test_acc))

    end_time = timer()
    load_time = end_time - start_time
    print()
    print('===== TOTAL TEST SET PREDICTION TIME: {0:.1f} sec ====='.format(load_time))
    print()




if __name__ == '__main__':
    app.run(main)
