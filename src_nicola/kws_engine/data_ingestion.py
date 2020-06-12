
import os

# LEO: disattivata la GPU, sembro avere un problema con i driver
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Mel features library
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import randrange

from absl import app

# stampa la GPU disponibile
# print(tf.config.experimental.list_physical_devices('GPU'))

# ---------------------------- PARAMETRI DI INPUT ----------------------------

TRAIN_DIR = "C:/Users/admin/Desktop/HDA/final_project/dataset/_"

# scegliere se usare come modello il classificatore di debug oppure l'autoencoder
DEBUG_CLASSIFIER = True

NUM_FEATURES = 26  # number of features per sample (Mel)
num_units = 5  # GRU units in encoder and decoder
BATCH_SIZE = 1
LR = 0.01
NUM_EPOCH = 2

# ----------------------------  FUNZIONI DI SUPPORTO ------------------------------

def debug_classifier_model(num_tokens, num_units, num_labels):
    """
    Modello di un classificatore RNN per il debug, classifica gli audio nell rispettiva classe.

    :param num_tokens:
    :param num_units:
    :return:
    """

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))

    # l'input è in forma [batch, timesteps, feature]
    encoder_outputs, encoder_state = tf.keras.layers.GRU(num_units, return_state=True)(encoder_inputs)

    fin_output = Dense(num_labels)(encoder_outputs)
    model = Model(encoder_inputs, fin_output)

    return model


def rnn_model(num_tokens, num_units):
    """
    Modello di autoencoder RNN, codifica e decodifica gli spettogrammi cercando di ricostruirli.

    :param num_tokens:
    :param num_units:
    :return:
    """

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder_outputs, encoder_state = tf.keras.layers.GRU(num_units, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    # TODO: rendere il decoder bidirezionale
    decoder_outputs, _ = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(decoder_inputs, initial_state = encoder_state)
    fin_output = Dense(num_tokens, activation=None)(decoder_outputs)

    seq_2_seq_autoencoder = Model((encoder_inputs, decoder_inputs), fin_output)
    encoder = Model(encoder_inputs, encoder_outputs)

    return seq_2_seq_autoencoder, encoder


def calculate_dec_input(enc_input):
    """
    Builds decoder input, as shifted encoder input

    :param enc_input:
    :return:
    """
    dec_input = np.zeros(enc_input.shape, dtype='float32')
    # dec_input(0) = [0 ... 0]; dec_input(T) = enc_input(T-1)

    #for sample in range(0, enc_input.shape[0]):
        #for timestep in range(0, enc_input.shape[1] - 1):
            #dec_input[sample, timestep + 1, :] = enc_input[sample, timestep, :]

    dec_input[1:, :] = enc_input[:-1, :]

    return dec_input


def compute_spectrogram(filename, num_filt):

    filename = filename.decode()
    #print(filename)
    rate, sig = wav.read(str(filename))

    # TODO: provare diversi valori per i paremtri di mfcc
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=0.01, numcep=num_filt,
                     nfilt=num_filt, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                     ceplifter=22, appendEnergy=True)

    # TODO: normalizzare i valori degli spettrogrammi? Ora sono circa in [-100, 100]

    spectr_pad = np.zeros((99, num_filt), dtype='float32')
    spectr_pad[:mfcc_feat.shape[0], :] = mfcc_feat

    return spectr_pad


def create_dataset(filenames, labels, num_filt, batch_size, shuffle, input_size, autoenc_mode, cache_file=None):
    """
    Crea un oggetto tf.data.Dataset da usare come input per un modello di classificazione o autoencoder

    :param filenames:
    :param labels:
    :param num_filt:
    :param batch_size:
    :param shuffle:
    :param input_size:
    :param autoenc_mode:
    :param cache_file:
    :return:
    """

    # Crea oggetti Dataset
    enc_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    target_dataset = tf.data.Dataset.from_tensor_slices(labels)

    # Mappa la funzione compute_spectrogram
    to_spectr = lambda filename: (tf.ensure_shape(tf.numpy_function(compute_spectrogram, [filename, num_filt],
                                                             tf.float32), input_size))

    enc_dataset = enc_dataset.map(to_spectr, num_parallel_calls=os.cpu_count())

    # modalità autoencoder, il target è l'input dell'encoder
    if autoenc_mode is True:
        # costruisci input del decoder
        # Mappa la funzione che azzera il primo timestep e shifta gli altri timesteps
        to_dec_input = lambda spectr: (tf.ensure_shape(tf.numpy_function(calculate_dec_input, [spectr],
                                                                        tf.float32), input_size))
        dec_dataset = enc_dataset.map(to_dec_input, num_parallel_calls=os.cpu_count())

        dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((enc_dataset, dec_dataset)), enc_dataset))

    # modalità classificazione, il target è la label
    else:
        # come scrivevo sotto, si potrebbe fare qui l'operazione di one hot encoding con qualcosa del tipo
        # dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda z: tf.one_hot(z, 10))
        dataset = tf.data.Dataset.zip((enc_dataset, target_dataset))

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # TODO: verificare lungo che asse dei dati viene eseguito lo shuffle -> Nicola: secondo me in automatico è ok, perchè nella doc non si parla mai di asse
    if shuffle:
        dataset = dataset.shuffle(len(filenames))

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


# ----------------------------  MAIN --------------------------

def main(argv):

    # LETTURA DEI FILENAME E CREAZIONE DELLE LABEL

    # # TODO: provare tf.Dataset.listfiles() per la lettura del dataset
    # # conta numero di file nel trainset
    # num_samples = 0
    # for subdirs, dirs, files in os.walk(TRAIN_DIR):
    #     # ad ogni loop, files contiene la lista dei filename presenti in una sottocartella
    #     # contiamo tutti i file che sono file audio wav
    #     files = [f for f in files if f.lower().endswith('.wav')]
    #     num_samples += len(files)
    # print('Files in the dataset: ' + str(num_samples))
    #
    # filenames = []
    # labels = np.zeros((num_samples, 1), dtype=int)
    # filenames_counter = 0
    # # il contatore delle label parte da -1 perchè itera sulle sottocartelle
    # # la directory indicata ha label -1 in quanto non contiene file ma cartelle
    # # la prima sottocartella (es. on) avrà label 0, la seguente 1, ecc.
    # labels_counter = -1
    #
    # for subdir, dirs, files in os.walk(TRAIN_DIR):
    #     for file in files:
    #         filepath = os.path.join(subdir, file)
    #
    #         if filepath.endswith(".wav"):
    #             filenames.append(filepath)
    #             labels[filenames_counter, 0] = labels_counter
    #             filenames_counter = filenames_counter + 1
    #
    #     # incrementa label numerica quando stiamo per passare alla prossima sottocartella
    #     labels_counter = labels_counter + 1
    #
    # # trasformazione della lista dei filename in numpy array
    # filenames_numpy = np.array(filenames)
    #
    # # trasformazione delle label in one hot encoding
    # labels_one_hot = tf.keras.utils.to_categorical(labels)


    filenames = []
    labels = []
    labels_counter = 0
    labels_dict = {}

    entry_list = os.listdir(TRAIN_DIR)
    entry_list.sort() # faccio così perchè os.listdir() restituisce in ordine arbitrario in teoria

    for entry in entry_list:

        # skipping files in root directory and background noise folder (non dovrebbe essere una classe ma era usata solo per aggiungere rumore mi sembra)
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
    labels_one_hot = tf.keras.utils.to_categorical(labels_numpy) # TODO questa operazione penso si possa fare al momento della
                                                                 # creazione del Dataset, esempio: dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda z: tf.one_hot(z, 10))

    # shuffling dei dati
    filenames_shuffled, labels_one_hot_shuffled = shuffle(filenames_numpy, labels_one_hot)

    X_train_filenames, X_val_filenames, Y_train, Y_val = train_test_split(
        filenames_shuffled, labels_one_hot_shuffled, test_size=0.05, random_state=1)

    # conversione dei vettori di label da float32 a int (negli step precendenti avviene la conversione, bisognerebbe scoprire dove)
    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    num_labels = Y_train.shape[1]

    print('Total number of audio files in the dataset: ' + str(filenames_numpy.shape[0]))
    print('Total number of classes in the dataset: ' + str(num_labels))
    print('Classes: ' + str(labels_dict.values()))
    print('Total number of audio files in the training set: ' + str(X_train_filenames.shape[0]))
    print('Total number of audio files in the validation set: ' + str(X_val_filenames.shape[0]))


    # CREAZIONE E TRAIN DEL MODELLO

    # steps per epoca in modo da passare tutto il dataset
    train_steps = int(np.ceil(X_train_filenames.shape[0] / BATCH_SIZE))
    val_steps = int(np.ceil(X_val_filenames.shape[0] / BATCH_SIZE))

    return

    # traina classificatore
    if DEBUG_CLASSIFIER:
        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False, (99, NUM_FEATURES), autoenc_mode=False)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False, (99, NUM_FEATURES), autoenc_mode=False)

        # crea e traina il modello con API Keras
        debug_classifier = debug_classifier_model(NUM_FEATURES, num_units, num_labels)
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        debug_classifier.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
        debug_classifier.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps, validation_data=val_dataset, validation_steps=val_steps)

    # traina autoencoder
    else:
        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False, (99, NUM_FEATURES), autoenc_mode=True)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False, (99, NUM_FEATURES), autoenc_mode=True)

        # crea e traina il modello con API Keras
        autoenc, _ = rnn_model(NUM_FEATURES, num_units)
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        autoenc.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])
        autoenc.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps, validation_data=val_dataset, validation_steps=val_steps)




if __name__ == '__main__':
    app.run(main)
