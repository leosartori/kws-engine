import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Dropout
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt

# Mel features library
from mfcc_base import mfcc
import scipy.io.wavfile as wav

# import re
# import hashlib
# MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


# ----------------------------  FUNZIONI DI SUPPORTO ------------------------------


# def debug_classifier_model(num_tokens, num_units, num_labels):
#     """
#     Modello di un classificatore RNN per il debug, classifica gli audio nell rispettiva classe.
#
#     :param num_tokens:
#     :param num_units:
#     :return:
#     """
#
#     # encoder
#     # Define an input sequence and process it.
#     encoder_inputs = Input(shape=(None, num_tokens))
#
#     # l'input è in forma [batch, timesteps, feature]
#     encoder_outputs, encoder_state = GRU(num_units, return_state=True)(encoder_inputs)
#
#     fin_output = Dense(num_labels)(encoder_outputs)
#     model = Model(encoder_inputs, fin_output)
#
#     return model


def rnn_autoencoder_model(num_features, num_units):
    """
    Modello di autoencoder RNN, codifica e decodifica gli spettogrammi cercando di ricostruirli.

    :param num_features:
    :param num_units:
    :return:
    """

    GRU_dropout_probability = 0.2 # as in the paper

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_features))

    # come descritto nel paper i layer dell'encoder hanno stato iniziale uguale a zero. Dovrebbe essere ok di default
    # documentazione tensorflow GRU: initial_state: List of initial state tensors to be passed to the first call of the cell
    #  (optional, defaults to None which causes creation of zero-filled initial state tensors).

    # N_l = 2 (due livelli)
    encoder_outputs_1, encoder_state_1 = GRU(num_units, return_sequences=True, return_state=True, dropout=GRU_dropout_probability)(encoder_inputs) # input = [batch, timesteps, feature]
    encoder_outputs_2, encoder_state_2 = GRU(num_units, return_state=True, dropout=GRU_dropout_probability)(encoder_outputs_1) # input = [batch, timesteps, feature]

    # encoder hidden states concatenation
    encoder_states_concatenated = tf.concat(values=[encoder_state_1, encoder_state_2], axis=1)
    # print(encoder_states_concatenated.shape)

    # # Dense Layer
    # encoder_state_1 = Dense(units=num_units, activation='relu')(encoder_state_1)
    # encoder_state_2 = Dense(units=num_units, activation='relu')(encoder_state_2)
    decoder_init_state = Dense(units=num_units, activation='tanh')(encoder_states_concatenated)
    decoder_init_state_list = [decoder_init_state, decoder_init_state]


    # Set up the decoder, using `encoder_states` as initial state.
    # init_dec_1 = (encoder_state_2, encoder_state_2)
    # init_dec_2 = (encoder_state_1, encoder_state_1)

    decoder_inputs = Input(shape=(None, num_features))

    # # N_l = 2 (due livelli)
    # decoder_outputs_1 = Bidirectional(
    #     GRU(num_units, return_sequences=True, dropout=GRU_dropout_probability))(decoder_inputs, initial_state=init_dec_1)
    # decoder_outputs_2 = Bidirectional(
    #     GRU(num_units, return_sequences=True, dropout=GRU_dropout_probability))(decoder_outputs_1, initial_state=init_dec_2)

    # N_l = 2 (due livelli)
    decoder_outputs_1 = Bidirectional(
        GRU(num_units, return_sequences=True, dropout=GRU_dropout_probability))(decoder_inputs, initial_state=decoder_init_state_list)
    decoder_outputs_2 = Bidirectional(
        GRU(num_units, return_sequences=True, dropout=GRU_dropout_probability))(decoder_outputs_1, initial_state=decoder_init_state_list)

    # dal paper: The weights of this output projection are shared across time steps
    # quindi così dovrebbe essere ok, mettendo TimeDistributed in teoria ottengo sharing nel tempo dei pesi
    # nel paper si parla di linear projection come layer finale...
    final_output = Dense(num_features, activation=None)(decoder_outputs_2)

    seq_2_seq_rnn_autoencoder = Model((encoder_inputs, decoder_inputs), final_output)

    return seq_2_seq_rnn_autoencoder



def rnn_encoder_mlp_model(rnn_autoencoder, num_units, num_classes, num_features):

    # transfer learning: https://keras.io/guides/transfer_learning/

    # encoder_model = Sequential()
    # encoder_model.add(rnn_autoencoder.layers[0]) # Input layer
    # encoder_model.add(rnn_autoencoder.layers[1]) # GRU RNN layer1
    # encoder_model.add(rnn_autoencoder.layers[2])  # GRU RNN layer2
    # encoder_model.add(rnn_autoencoder.layers[3])  # State Concatenation layer
    # encoder_model.add(rnn_autoencoder.layers[5])  # State Dense layer

    # print("Descrizione del modello per il fine tuning dopo il training dell'autoencoder")
    #
    # print(rnn_autoencoder.layers[0].input)
    # print(rnn_autoencoder.layers[1].input)
    # print(rnn_autoencoder.layers[2].input)
    # print(rnn_autoencoder.layers[3].input)
    #
    # print(rnn_autoencoder.layers[0].output)
    # print(rnn_autoencoder.layers[1].output)
    # print(rnn_autoencoder.layers[2].output)
    # print(rnn_autoencoder.layers[3].output)

    encoder_inputs = Input(shape=(None, num_features))
    # encoder_inputs = rnn_autoencoder.layers[0](input)
    # encoder_rnn_layer1 = rnn_autoencoder.layers[1](encoder_inputs.output)
    encoder_rnn_layer1 = rnn_autoencoder.layers[1](encoder_inputs)
    encoder_rnn_layer2 = rnn_autoencoder.layers[2](encoder_rnn_layer1[0])

    # print()
    # print(type(encoder_rnn_layer1[0]))
    # print(type(encoder_rnn_layer1[1]))
    # print(type(encoder_rnn_layer2[0]))
    # print(type(encoder_rnn_layer2[1]))

    encoder_states_concatenation = rnn_autoencoder.layers[3]((encoder_rnn_layer1[1], encoder_rnn_layer2[1]))

    encoder_dense = rnn_autoencoder.layers[5](encoder_states_concatenation)

    encoder_model = Model(encoder_inputs, encoder_dense)

    encoder_model.trainable = False


    dropout_probability = 0.4 # as in the paper

    classification_model = Sequential()
    classification_model.add(encoder_model)
    classification_model.add(Dropout(dropout_probability))
    classification_model.add(Dense(num_units, activation='relu'))
    classification_model.add(Dropout(dropout_probability))
    classification_model.add(Dense(num_units, activation='relu'))
    classification_model.add(Dropout(dropout_probability))
    classification_model.add(Dense(num_classes, activation='softmax'))

    return classification_model, encoder_model




def calculate_dec_input(enc_input):
    """
    Builds decoder input, as shifted encoder input

    :param enc_input:
    :return:
    """
    # dec_input(0) = [0... 0]; dec_input(T) = enc_input(T - 1)

    # inizializza vettore di output
    dec_input = np.zeros(enc_input.shape, dtype='float32')

    # copia dai valori di input encoder ma shiftati di un timestep
    # dec_input[1:, :] = enc_input[:-1, :]

    # In order to introduce greater short - term dependencies between the encoder and the decoder,
    # our ecoder RNN reconstructs the reversed input sequence
    dec_input_tmp = np.flip(enc_input, axis=0)
    dec_input[1:, :] = dec_input_tmp[:-1, :]

    return dec_input



def calculate_dec_output(enc_output):
    # In order to introduce greater short - term dependencies between the encoder and the decoder,
    # our ecoder RNN reconstructs the reversed input sequence
    return np.flip(enc_output, axis=0)



# # funzione presa da https://github.com/jameslyons/python_speech_features/commit/9ab32879b1fb31a38c1a70392fd21370b8fdc30f
# # (commit nella repo di python_speech_features), serve per calcolare il parametro nfft automaticamente da rate e winlen
# # Dovrebbe funzionare ugualmente mettendo nfft=None come parametro della funzione mfcc, ma a me dava errore
# def calculate_nfft(samplerate, winlen):
#     """Calculates the FFT size as a power of two greater than or equal to
#     the number of samples in a single window length.
#
#     Having an FFT less than the window length loses precision by dropping
#     many of the samples; a longer FFT than the window allows zero-padding
#     of the FFT buffer which is neutral in terms of frequency domain conversion.
#     :param samplerate: The sample rate of the signal we are working with, in Hz.
#     :param winlen: The length of the analysis window in seconds.
#     """
#     window_length_samples = winlen * samplerate
#     nfft = 1
#     while nfft < window_length_samples:
#         nfft *= 2
#     return nfft


def compute_spectrogram(filename, num_features, win_len, win_step):

    filename = filename.decode()
    rate, signal = wav.read(str(filename))

    # siccome sappiamo che la durata massima di un file audio è un secondo (cioè max(length(signal)) = rate)
    # tengo al massimo un secondo
    signal = signal[0:rate]
    # faccio padding con zeri fino a 1 secondo se l'audio è più corto
    signal_padded = np.pad(signal, (0, rate - signal.shape[0]))

    # print('Rate: ' + str(rate))
    # print('Signal length: ' + str(len(sig)))

    mfcc_features = mfcc(signal_padded, samplerate=rate, winlen=win_len, winstep=win_step, numcep=num_features,
                     nfilt=num_features, nfft=512, #calculate_nfft(rate, win_len),
                     lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)

    # print(mfcc_feat.shape)

    # TODO: normalizzare i valori degli spettrogrammi come nel paper Seq-to-Seq (par 2.1)? Ora sono circa in [-100, 100]
    # domanda: da paper non si capisce bene (o almeno io non capisco bene) come venga fatta quella normalizzazione tra -1 e -1
    # cioè non capisco se venga fatta individualmente per ogni spettrogramma oppure se venga fatta sul tensore contato come un insieme
    # a questo punto proverei a lasciare la normalizzazione fatta con la funzione di tensorflow,
    # anche se non penso proprio che normalizzi tra -1 e 1

    # faccio padding a
    # spectr = np.zeros((mfcc_feat.shape[0], num_filt), dtype='float32')
    # spectr[:, :] = mfcc_feat
    # return spectr

    # spectr_pad[:,:] = mfcc_feat

    return mfcc_features.astype(dtype='float32')



def normalize_tensor(x, axes=[0], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
    return x_normed


def create_dataset(filenames, labels, num_features, batch_size, shuffle, input_size, network_model,
                   win_len, win_step, normalize=False, cache_file=None, mode='train'):
    """
    Crea un oggetto tf.data.Dataset da usare come input per un modello di classificazione o autoencoder

    :param filenames:
    :param labels:
    :param num_features:
    :param batch_size:
    :param shuffle:
    :param input_size:
    :param autoenc_mode:
    :param cache_file:
    :return:
    """

    # Crea oggetti Dataset
    input_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    target_dataset = tf.data.Dataset.from_tensor_slices(labels)

    # Mappa la funzione compute_spectrogram
    to_spectrogram = lambda filename: (tf.ensure_shape(tf.numpy_function(compute_spectrogram,
                                                                    [filename, num_features, win_len, win_step],
                                                                    tf.float32), input_size))

    dataset = input_dataset.map(to_spectrogram, num_parallel_calls=os.cpu_count())

    if normalize:
        train_dataset = dataset.map(normalize_tensor, num_parallel_calls=os.cpu_count())
    else:
        train_dataset = dataset


    # autoencoder, il target è l'input dell'encoder
    if network_model == 'autoencoder1':
        # costruisci input del decoder
        # Mappa la funzione che azzera il primo timestep e shifta gli altri timesteps
        to_dec_input = lambda spectr: (tf.ensure_shape(tf.numpy_function(calculate_dec_input, [spectr],
                                                                        tf.float32), input_size))
        dec_in_dataset = train_dataset.map(to_dec_input, num_parallel_calls=os.cpu_count())

        # dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((norm_enc_dataset, dec_in_dataset)), norm_enc_dataset))

        # In order to introduce greater short - term dependencies between the encoder and the decoder,
        # our ecoder RNN reconstructs the reversed input sequence
        to_dec_output = lambda spectr: (tf.ensure_shape(tf.numpy_function(calculate_dec_output, [spectr],
                                                                         tf.float32), input_size))
        dec_out_dataset = train_dataset.map(to_dec_output, num_parallel_calls=os.cpu_count())

        dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((train_dataset, dec_in_dataset)), dec_out_dataset))


    # modalità classificazione, il target è la label
    if network_model == 'debug_classifier' or network_model == 'encoder_mlp_classifier1':
        # come scrivevo sotto, si potrebbe fare qui l'operazione di one hot encoding con qualcosa del tipo
        # dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda z: tf.one_hot(z, 10))
        dataset = tf.data.Dataset.zip((train_dataset, target_dataset))



    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # TODO: verificare lungo che asse dei dati viene eseguito lo shuffle -> Nicola: secondo me in automatico è ok, perchè nella doc non si parla mai di asse
    # secondo me funziona però necessita molto tempo... da provare nel cluster
    if shuffle:
        dataset = dataset.shuffle(len(filenames))

    # Repeat the dataset indefinitely only during training (and not during testing phase)
    if mode == 'train':
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
        return float(alpha)



def split_dataset_from_list(filename_list, labels_list, validation_file, testing_file):
    """Determines which data partition the file should belong to using the provided .txt files and creates
    three list of filenames respectively for training, validation and test set.
    """

    train_set_filename = []
    train_set_label = []
    val_set_filename = []
    val_set_label = []
    test_set_filename = []
    test_set_label = []

    # open files reporting the division between validation set and test set
    val_txt = open(validation_file, "r").read()
    test_txt = open(testing_file, "r").read()

    val_list = val_txt.split('\n')[:-1]
    test_list = test_txt.split('\n')[:-1]


    for filename, label in zip(filename_list, labels_list):

        # i file di testo contengono i filename in formato: right/a69b9b3e_nohash_0.wav
        # mentre noi qui in filename abbiamo il path completo, ne prendo solo la parte finale
        filename_path_split = filename.split('/')
        filename_no_path = filename_path_split[-2] + '/' + filename_path_split[-1]

        if filename_no_path in val_list:
            val_set_filename.append(filename)
            val_set_label.append(label)
        elif filename_no_path in test_list:
            test_set_filename.append(filename)
            test_set_label.append(label)
        else:
            train_set_filename.append(filename)
            train_set_label.append(label)

    return train_set_filename, train_set_label, val_set_filename, val_set_label, test_set_filename, test_set_label



def save_training_loss_trend_plot(history, network_model, model_version, loss_type):

    # list all data in history
    # print(history.history.keys())

    # plot loss (mse)
    plt.title('Loss Function - ' + loss_type)
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    # save plot to file
    filename = './training_output/images/loss-resume_' + network_model + '_v' + str(model_version) + '.png'
    plt.savefig(filename)
    plt.close()


def save_training_accuracy_trend_plot(history, network_model, model_version):

    # list all data in history
    # print(history.history.keys())

    # plot accuracy
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    # save plot to file
    filename = './training_output/images/acc-resume_' + network_model + '_v' + str(model_version) + '.png'
    plt.savefig(filename)
    plt.close()


def printInfo(network_model_to_train, model_version_to_train, num_features, batch_size, max_timesteps_stectrograms, win_len, win_step, num_epochs):

    print()
    print('TRAINING INFORMATION:')
    print('Network model to train: ' + network_model_to_train)
    print('Model version to train: ' + str(model_version_to_train))
    print('Number of Mel features: ' + str(num_features))
    print('Spectrograms timesteps: ' + str(max_timesteps_stectrograms))
    print('Time length of each frame: ' + str(win_len))
    print('Time step/shift of each frame: ' + str(win_step))
    print('Batch size: ' + str(batch_size))
    print('Number of epochs: ' + str(num_epochs))
    print()