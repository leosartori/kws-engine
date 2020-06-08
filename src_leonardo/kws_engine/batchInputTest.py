import os
# LEO: disattivata la GPU, sembro avere un problema con i driver
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# produce Mel features
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import randrange

# stampa la GPU disponibile
# print(tf.config.experimental.list_physical_devices('GPU'))

TRAIN_DIR = "C:\\Users\\Leonardo\\Documents\\Uni\\HDA\\Project\\debug_dataset_020620\\train"


def rnn_model(num_tokens, num_units):
    """
    Creates the RNN model

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

    seq_2_seq_autoencoder = Model([encoder_inputs, decoder_inputs], fin_output)
    encoder = Model(encoder_inputs, encoder_outputs)

    return seq_2_seq_autoencoder, encoder


def calculate_dec_input(enc_input):
    """
    Builds decoder input, as shifted encoder input

    :param enc_input:
    :return:
    """
    dec_input = np.zeros(enc_input.shape)
    # dec_input(0) = [0 ... 0]; dec_input(T) = enc_input(T-1)
    for sample in range(0, enc_input.shape[0]):
        for timestep in range(0, enc_input.shape[1] - 1):
            dec_input[sample, timestep + 1, :] = enc_input[sample, timestep, :]
    return dec_input

# NON PIU' UTILIZZATO: questa funzione calcola una batch di spettogrammi randomica
# il problema è che per fare un'epoca di train, dobbiamo passare tutto il dataset diviso in batch
# e non un numero N di batch randomiche (che probabilisticamente ripetono alcuni file e ne escludono altri)
def get_data(filenames_list, labels_list, mode='train', batch_size=32, num_filt=26):
    """
    NON PIU' UTILIZZATO
    Retrieves a complete set of batches of encoder input data, decoder input data and labels

    :param filenames_list:
    :param labels_list:
    :param mode:
    :param batch_size:
    :param num_filt:
    :return:
    """
    if mode == 'train':
        # shuffling dei file di train per avere sempre batch differenti
        filenames_list, labels_list = shuffle(filenames_list, labels_list)

        # get batch of filenames and labels
        rand_index = randrange(len(filenames_list) - batch_size)
        batch_filename = filenames_list[rand_index: rand_index + batch_size]
        batch_y = labels_list[rand_index: rand_index + batch_size]
    elif mode == 'eval' or mode == 'test':
        # in caso di evaluation o test mantieniamo non facciamo shuffling e batching
        batch_filename = filenames_list
        batch_y = labels_list
    else:
        raise Exception('Mode for get_data function must be train, eval or test')

    batch_x = compute_spectrograms(batch_filename, num_filt)
    # shiftare la batch di file di training per creare l'input del decoder
    batch_x_shifted = calculate_dec_input(batch_x)
    batch_y = np.array(batch_y)

    # obtain input data for decoder train as shifted encoder data
    print('Creata batch: ' + str(batch_x.shape))

    return batch_x, batch_x_shifted, batch_y


def random_mini_batches(X, Y, mini_batch_size=32, seed=0):
    """
    Groups random minibatches from (X, Y) to perform an epoch of training and returns the list of the groups

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def compute_spectrograms(filenames, num_filt):
    """
    Compute the spectrograms for a list of filenames
    :param filenames:
    :param num_filt:
    :return:
    """

    num_samples = len(filenames)
    spectr = np.zeros((num_samples, 99, num_filt))  # LEO: 99 è il numero massimo di timestep che mfcc ottiene per il
    # dataset speech_commands, ma non è stabile (ad es. da
    # on\0a9f9af7_nohash_1.wav ho 84 timestep) perchè?

    sample_id = 0
    # genera batch di input per encoder
    for filename in filenames:
        rate, sig = wav.read(str(filename))
        # TODO: provare diversi valori per i paremtri di mfcc
        mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=0.01, numcep=num_features,
                         nfilt=num_filt, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                         ceplifter=22, appendEnergy=True)
        # TODO: normalizzare i valori degli spettrogrammi? Ora sono circa in [-100, 100]
        spectr[sample_id, :mfcc_feat.shape[0], :] = mfcc_feat  # LEO: il match della seconda dimensione è dovuto al commento sopra
        sample_id += 1

    return spectr


def compute_cost(y_pred, y_true):
    """
    Computes the cost function using MSE

    :param y_pred:
    :param y_true:
    :return:
    """
    cost = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

    return cost


# TODO: gestisci possibilità di avere una lista di num_epoche e una lista di learning_rate
def train_autoencoder(num_features, num_units, X_train_filenames, Y_train, X_test_filenames, Y_test, learning_rate=0.1,
          num_epochs=1, minibatch_size=32, print_cost=True):
    """
    Performs the training of the autoencoder model
    :param num_features:
    :param num_units:
    :param X_train_filenames:
    :param Y_train:
    :param X_test_filenames:
    :param Y_test:
    :param learning_rate:
    :param num_epochs:
    :param minibatch_size:
    :param print_cost:
    :return:
    """
    network_model, encoder_model = rnn_model(num_features, num_units)

    tf.random.set_seed(1)  # to keep results consistent (tensorflow seed)
    num_samples = X_train_filenames.shape[0]
    seed = 3  # to keep results consistent (numpy seed), to pass to get_data
    costs = []  # To keep track of the cost

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # parametro usato per cambiare il learning rate durante il training
    lr_change = False

    # Do the training loop
    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(num_samples / minibatch_size)  # number of minibatches of size minibatch_size in the train set
        seed += 1

        # LEO: random_mini_batches restituisce come batch una lista di filenames e una di label relative
        # le label non ci sono necessarie nel training dell'autoencoder ma lo saranno nella classificazione
        minibatches = random_mini_batches(X_train_filenames, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            with tf.GradientTape() as tape:
                # Select a minibatch
                (minibatch_X_filenames, minibatch_Y) = minibatch
                # calculate spectrograms
                # TODO: verificare che gli spettrogrammi calcolati per ogni minibatch sovrascrivano quelli della minibatch precedente, rimuovendoli dalla RAM
                minibatch_X_spectr_enc = compute_spectrograms(minibatch_X_filenames, num_features)
                minibatch_X_spectr_dec = calculate_dec_input(minibatch_X_spectr_enc)

                # Forward propagation
                output_pred = network_model([minibatch_X_spectr_enc, minibatch_X_spectr_dec])

                # Cost function for the autoencoder
                cost = compute_cost(output_pred, minibatch_X_spectr_enc)

            # Compute the gradient
            gradients = tape.gradient(cost, network_model.trainable_variables)

            # Apply the optimizer
            optimizer.apply_gradients(zip(gradients, network_model.trainable_variables))
            minibatch_cost += cost / num_minibatches

        # Print the cost every epoch
        if print_cost is True and epoch % 1 == 0:
            print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost is True and epoch % 1 == 0:
            costs.append(minibatch_cost)

        # soluzione temporanea per variare il learning rate dopo 9/10 delle epoche totali
        if epoch > (num_epochs * (9/10)) and lr_change is False:
            print('Learning rate changed on epoch: ' + str(epoch))
            learning_rate /= 10
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            lr_change = True

    # Calcola prestazioni del modello sul testset
    test_X_spectr_enc = compute_spectrograms(X_test_filenames, num_features)
    test_X_spectr_dec = calculate_dec_input(test_X_spectr_enc)

    # Forward propagation
    output_pred_test = network_model([test_X_spectr_enc, test_X_spectr_dec])

    # Cost function for the autoencoder
    test_cost = compute_cost(output_pred_test, test_X_spectr_enc).numpy()
    print("Cost on test:", test_cost)

    # ritorna i modelli di autoencoder e encoder
    return network_model, encoder_model, minibatch_cost


# LETTURA DEL TRAINSET
# conta numero di file nel trainset
num_samples = 0
for subdirs, dirs, files in os.walk(TRAIN_DIR):
    # ad ogni loop, files contiene la lista dei filename presenti in una sottocartella
    # contiamo tutti i file che sono file audio wav
    files = [f for f in files if f.lower().endswith('.wav')]
    num_samples += len(files)
print('Files in the dataset: ' + str(num_samples))

filenames = []
labels = np.zeros((num_samples, 1), dtype=int)
filenames_counter = 0
# il contatore delle label parte da -1 perchè itera sulle sottocartelle
# la directory indicata ha label -1 in quanto non contiene file ma cartelle
# la prima sottocartella (es. on) avrà label 0, la seguente 1, ecc.
labels_counter = -1

for subdir, dirs, files in os.walk(TRAIN_DIR):
    for file in files:
        filepath = os.path.join(subdir, file)

        if filepath.endswith(".wav"):
            filenames.append(filepath)
            labels[filenames_counter, 0] = labels_counter
            filenames_counter = filenames_counter + 1

    # incrementa label numerica quando stiamo per passare alla prossima sottocartella
    labels_counter = labels_counter + 1

# trasformazione della lista dei filename in numpy array
filenames_numpy = np.array(filenames)

# trasformazione delle label in one hot encoding
labels_one_hot = tf.keras.utils.to_categorical(labels)

# shuffling dei dati
filenames_shuffled, labels_one_hot_shuffled = shuffle(filenames_numpy, labels_one_hot)

X_train_filenames, X_test_filenames, Y_train, Y_test = train_test_split(
    filenames_shuffled, labels_one_hot_shuffled, test_size=0.05, random_state=1)

# CREAZIONE E TRAIN DEL MODELLO
num_features = 26  # number of features per sample (Mel)
num_units = 5  # GRU units in encoder and decoder

print("Labels in the dataset: " + str(Y_train.shape[1]))
print("Files in the trainset: " + str(X_train_filenames.shape[0]))
print("Files in the testset: " + str(X_test_filenames.shape[0]))

batch_size = 32

autoencoder, encoder, batch_cost_list = train_autoencoder(num_features, num_units, X_train_filenames, Y_train,
                                         X_test_filenames, Y_test, num_epochs=10, minibatch_size=batch_size)
