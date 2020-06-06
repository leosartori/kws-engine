import os
import shutil
# LEO: disattivata la GPU, sembro avere un problema con i driver
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# produce Mel features
from python_speech_features import mfcc
import scipy.io.wavfile as wav

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import randrange

# stampa la GPU disponibile
print(tf.config.experimental.list_physical_devices('GPU'))

TRAIN_DIR = "C:\\Users\\Leonardo\\Documents\\Uni\\HDA\\Project\\debug_dataset_020620"

def rnn_model(num_tokens, num_units):

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder_outputs, encoder_state = tf.keras.layers.GRU(num_units, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    decoder_outputs, _ = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(decoder_inputs, initial_state = encoder_state)
    fin_output = Dense(num_tokens, activation=None)(decoder_outputs)

    seq_2_seq_autoencoder = Model([encoder_inputs, decoder_inputs], fin_output)
    encoder = Model(encoder_inputs, encoder_outputs)

    return seq_2_seq_autoencoder, encoder


def calculate_dec_input(enc_input):
    ''' Build decoder input, as shifted encoder input '''
    dec_input = np.zeros(enc_input.shape)
    # dec_input(0) = [0 ... 0]; dec_input(T) = enc_input(T-1)
    for sample in range(0, enc_input.shape[0]):
        for timestep in range(0, enc_input.shape[1] - 1):
            dec_input[sample, timestep + 1, :] = enc_input[sample, timestep, :]
    return dec_input


def get_data(filenames_list, labels_list, mode='train', batch_size=32, num_filt=26):
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

    num_samples = len(batch_filename)
    batch_x = np.zeros((num_samples, 99, num_filt)) # LEO: 99 è il numero massimo di timestep che mfcc ottiene, ma non è stabile (alcune volte è a 84, 72...) perchè?

    sample_id = 0
    # generate batch of encoder data
    for file_name in batch_filename:
        rate, sig = wav.read(file_name)
        mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=0.01, numcep=num_features,
                         nfilt=num_filt, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                         ceplifter=22, appendEnergy=True)
        print(mfcc_feat.shape)
        batch_x[sample_id, :mfcc_feat.shape[0], :] = mfcc_feat # LEO: il match della seconda dimensione è dovuto al commento sopra

    # obtain input data for decoder train as shifted encoder data
    print(batch_x.shape)

    # shiftare la batch di file di training per creare l'input del decoder
    batch_x_shifted = calculate_dec_input(batch_x)
    batch_y = np.array(batch_y)

    # TODO: the element generated is only suitable for autoencoder train, we must change it for audio classification
    return batch_x, batch_x_shifted, batch_y


# LETTURA DEL DATASET
# count number of samples in the dataset
num_samples = 0
for subdirs, dirs, files in os.walk(TRAIN_DIR):
    num_samples += len(files)
print('Files in the dataset: ' + str(num_samples))

filenames = []
labels = np.zeros((num_samples, 1))
filenames_counter = 0
labels_counter = -1

for subdir, dirs, files in os.walk(TRAIN_DIR):
    for file in files:
        filepath = os.path.join(subdir, file)

        if filepath.endswith(".wav"):
            # print(filepath)
            filenames.append(filepath)
            labels[filenames_counter, 0] = labels_counter
            filenames_counter = filenames_counter + 1

    # assign numeric index based on the directory
    labels_counter = labels_counter + 1

print(len(filenames))
print(labels.shape)

# trasformazione delle label in one hot encoding
labels_one_hot = tf.keras.utils.to_categorical(labels)
# shuffling dei dati
filenames_shuffled, labels_one_hot_shuffled = shuffle(filenames, labels_one_hot)

filenames_shuffled_numpy = np.array(filenames_shuffled)

X_train_filenames, X_val_filenames, Y_train, Y_val = train_test_split(
    filenames_shuffled_numpy, labels_one_hot_shuffled, test_size=0.05, random_state=1)

# CREAZIONE E TRAIN DEL MODELLO
num_features = 26  # number of features per sample (Mel)
num_units = 5  # GRU units in encoder and decoder
autoenc, enc = rnn_model(num_features, num_units)

# ottiene il dataset
X_train_batch, X_train_shifted_batch, labels = get_data(X_train_filenames, Y_train,
                                                  mode='train', batch_size=32, num_filt=num_features)

# train the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
autoenc.compile(optimizer = opt, loss = tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.AUC()])

# al momento, per testare la funzione di generazione batch
# genero una singola batch e la uso come trainset (completo) nella funzione fit di Keras
# in futuro andrà usato in un loop di tensorflow
autoenc.fit(x = [X_train_batch, X_train_shifted_batch], y=X_train_batch, epochs=2, batch_size = 34)
