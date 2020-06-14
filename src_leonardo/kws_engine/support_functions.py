import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional
from tensorflow.keras.models import Model

# Mel features library
from mfcc_base import mfcc
import scipy.io.wavfile as wav

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
    encoder_outputs, encoder_state = GRU(num_units, return_state=True)(encoder_inputs)

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
    encoder_outputs_1, encoder_state_1 = GRU(num_units, return_sequences=True, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]
    encoder_outputs_2, encoder_state_2 = GRU(num_units, return_state=True)(encoder_outputs_1) # input = [batch, timesteps, feature]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    init_dec_1 = (encoder_state_2, encoder_state_2)
    init_dec_2 = (encoder_state_1, encoder_state_1)
    # N_l = 2 (due livelli)
    decoder_outputs_1, *_ = Bidirectional(
        GRU(num_units, return_sequences=True, return_state=True))(decoder_inputs, initial_state=init_dec_1)
    decoder_outputs_2, *_ = Bidirectional(
        GRU(num_units, return_sequences=True, return_state=True))(decoder_outputs_1, initial_state=init_dec_2)

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


def compute_spectrogram(filename, num_filt, max_timesteps, win_len, win_step):

    filename = filename.decode()
    rate, sig = wav.read(str(filename))

    # TODO: provare diversi valori per i paremtri di mfcc
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=win_len, winstep=win_step, numcep=num_filt,
                     nfilt=num_filt, nfft=calculate_nfft(rate, win_len), lowfreq=0, highfreq=None, preemph=0.97,
                     ceplifter=22, appendEnergy=True)

    # TODO: normalizzare i valori degli spettrogrammi come nel paper Seq-to-Seq (par 2.1)? Ora sono circa in [-100, 100]

    spectr_pad = np.zeros((max_timesteps, num_filt), dtype='float32')
    # il padding è dovuto al fatto che non da tutti i sample mfcc genera lo stesso numero di timesteps (nonostante
    # siamo tutti file da 1 sec): la maggior parte sono lunghi 9, ma alcuni 8,7 o 6. PERCHE'?
    spectr_pad[:mfcc_feat.shape[0], :] = mfcc_feat

    return spectr_pad


def create_dataset(filenames, labels, num_filt, batch_size, shuffle, input_size, autoenc_mode,
                   max_timesteps, win_len, win_step, cache_file=None):
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
    to_spectr = lambda filename: (tf.ensure_shape(tf.numpy_function(compute_spectrogram,
                                                                    [filename, num_filt, max_timesteps, win_len, win_step],
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


# vedi https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
class StepDecay:
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor(epoch / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        # return the learning rate
        print('LR=' + str(alpha))

        return float(alpha)
