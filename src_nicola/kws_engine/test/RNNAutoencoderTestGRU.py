import os
# LEO: disattivata la GPU, sembro avere un problema con i driver
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# produce Mel features
from python_speech_features import mfcc
import scipy.io.wavfile as wav

# stampa la GPU disponibile
print(tf.config.experimental.list_physical_devices('GPU'))

# Within a single batch, you must have the same number of timesteps (this is typically where you see 0-padding and masking).
# But between batches there is no such restriction. During inference, you can have any length.
# (https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras)

def rnn_model(num_tokens, num_units):

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder_outputs, encoder_state = tf.keras.layers.GRU(num_units, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]

    print(encoder_outputs.shape)
    print(encoder_state.shape)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    decoder_outputs, _ = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(decoder_inputs, initial_state = encoder_state)
    fin_output = Dense(num_tokens, activation=None)(decoder_outputs)

    print(fin_output.shape)

    seq_2_seq_autoencoder = Model([encoder_inputs, decoder_inputs], fin_output)
    encoder = Model(encoder_inputs, encoder_outputs)

    return seq_2_seq_autoencoder, encoder


# create the model
num_features = 26 # number of features per sample (Mel)
num_units = 5 # GRU units in encoder and decoder
autoenc, enc = rnn_model(num_features, num_units)

# creating a fake trainset set with 10 samples, just repeating one sample wav file
X_train = []
for i in range (0, 100):
    (rate, sig) = wav.read("data/0a196374_nohash_0.wav")
    mfcc_feat = mfcc(sig, samplerate=rate,winlen=0.025,winstep=0.01,numcep=num_features,
                     nfilt=num_features,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
                     ceplifter=22,appendEnergy=True) # capire bene come funziona questa funzione...
    print(mfcc_feat.shape)
    X_train.append(mfcc_feat)
X_train = np.array(X_train)

print(X_train.shape)

# decoder input, as shifted encoder input
X_train_shifted = np.zeros(X_train.shape)
# loop in timesteps
for sample in range(0, X_train.shape[0]):
    for timestep in range(0, X_train.shape[1] - 1):
        X_train_shifted[sample, timestep + 1, :] = X_train[sample, timestep, :]

# train the model
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
autoenc.compile(optimizer = opt, loss = tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.MeanSquaredError()])
autoenc.fit(x = [X_train, X_train_shifted], y=X_train, epochs=20000, batch_size = 34)

# link utili:
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/lstm_seq2seq/
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://keras.io/api/layers/recurrent_layers/lstm/
# https://machinelearningmastery.com/lstm-autoencoders/
