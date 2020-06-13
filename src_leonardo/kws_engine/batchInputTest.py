
import os

# LEO: disattivata la GPU, sembro avere un problema con i driver
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Mel features library
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from absl import app

# stampa la GPU disponibile
# print(tf.config.experimental.list_physical_devices('GPU'))

# ---------------------------- PARAMETRI DI INPUT ----------------------------

TRAIN_DIR = "C:/Users/Leonardo/Documents/Uni/HDA/Project/debug_dataset_020620/train"

# scegliere se usare come modello il classificatore di debug oppure l'autoencoder
DEBUG_CLASSIFIER = False

NUM_FEATURES = 320  # number of features per sample (Mel)
NUM_UNITS = 256  # GRU units in encoder and decoder
BATCH_SIZE = 32
LR = 0.01
NUM_EPOCH = 100

MAX_LENGTH_TIMESTEPS = 9
WIN_LEN = 0.2
WIN_STEP = 0.1

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
    # N_l = 2 (due livelli)
    encoder_outputs_1, encoder_state_1 = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]
    encoder_outputs_2, encoder_state_2 = tf.keras.layers.GRU(num_units, return_state=True)(encoder_outputs_1) # input = [batch, timesteps, feature]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    init_dec_1 = (encoder_state_2, encoder_state_2)
    init_dec_2 = (encoder_state_1, encoder_state_1)
    # N_l = 2 (due livelli)
    decoder_outputs_1, *_ = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True))(decoder_inputs, initial_state=init_dec_1)
    decoder_outputs_2, *_ = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True))(decoder_outputs_1, initial_state=init_dec_2)

    fin_output = Dense(num_tokens, activation=None)(decoder_outputs_2)

    seq_2_seq_autoencoder = Model((encoder_inputs, decoder_inputs), fin_output)
    encoder = Model(encoder_inputs, encoder_outputs_2)

    return seq_2_seq_autoencoder, encoder


def calculate_dec_input(enc_input):
    """
    Builds decoder input, as shifted encoder input

    :param enc_input:
    :return:
    """
    # dec_input(0) = [0... 0]; dec_input(T) = enc_input(T - 1)

    # inizializza vettore di otput
    dec_input = np.zeros(enc_input.shape, dtype='float32')
    # copia dai valori di input encoder ma shiftati di un timestep
    dec_input[1:, :] = enc_input[:-1, :]

    return dec_input


# funzione presa da https://github.com/jameslyons/python_speech_features/commit/9ab32879b1fb31a38c1a70392fd21370b8fdc30f
# (commit nella repo di python_speech_features), serve per calcolare il parametro nfft automaticamente da rate e winlen
# Dovrebbe funzionare ugualmente mettendo nfft=None come parametro della funzione mfcc, ma a me dava errore
def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.

    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def compute_spectrogram(filename, num_filt):

    filename = filename.decode()
    rate, sig = wav.read(str(filename))

    # TODO: provare diversi valori per i paremtri di mfcc
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=WIN_LEN, winstep=WIN_STEP, numcep=num_filt,
                     nfilt=num_filt, nfft=calculate_nfft(rate, WIN_LEN), lowfreq=0, highfreq=None, preemph=0.97,
                     ceplifter=22, appendEnergy=True)

    # TODO: normalizzare i valori degli spettrogrammi come nel paper Seq-to-Seq (par 2.1)? Ora sono circa in [-100, 100]

    spectr_pad = np.zeros((MAX_LENGTH_TIMESTEPS, num_filt), dtype='float32')
    # il padding è dovuto al fatto che non da tutti i sample mfcc genera lo stesso numero di timesteps (nonostante
    # siamo tutti file da 1 sec): la maggior parte sono lunghi 9, ma alcuni 8,7 o 6. PERCHE'?
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

    # traina classificatore
    if DEBUG_CLASSIFIER:
        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False,
                                       (MAX_LENGTH_TIMESTEPS, NUM_FEATURES), autoenc_mode=False)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False,
                                     (MAX_LENGTH_TIMESTEPS, NUM_FEATURES), autoenc_mode=False)

        # crea e traina il modello con API Keras
        debug_classifier = debug_classifier_model(NUM_FEATURES, NUM_UNITS, num_labels)
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        debug_classifier.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
        debug_classifier.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps, validation_data=val_dataset, validation_steps=val_steps)

    # traina autoencoder
    else:
        # crea dataset con classe Dataset di TF
        train_dataset = create_dataset(X_train_filenames, Y_train, NUM_FEATURES, BATCH_SIZE, False,
                                       (MAX_LENGTH_TIMESTEPS, NUM_FEATURES), autoenc_mode=True)
        val_dataset = create_dataset(X_val_filenames, Y_val, NUM_FEATURES, BATCH_SIZE, False,
                                     (MAX_LENGTH_TIMESTEPS, NUM_FEATURES), autoenc_mode=True)

        # crea e traina il modello con API Keras
        autoenc, _ = rnn_model(NUM_FEATURES, NUM_UNITS)
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        autoenc.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])
        autoenc.fit(x=train_dataset, epochs=NUM_EPOCH, steps_per_epoch=train_steps, validation_data=val_dataset, validation_steps=val_steps)


if __name__ == '__main__':
    app.run(main)
