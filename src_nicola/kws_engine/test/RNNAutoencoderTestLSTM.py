
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# produce Mel features
from python_speech_features import mfcc
import scipy.io.wavfile as wav


# Within a single batch, you must have the same number of timesteps (this is typically where you see 0-padding and masking).
# But between batches there is no such restriction. During inference, you can have any length.
# (https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras)

def rnn_model(num_tokens, num_units):

    # encoder
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(num_tokens, return_state=True)(encoder_inputs) # input = [batch, timesteps, feature]

    print(encoder_outputs.shape)
    print(state_h.shape)
    print(state_c.shape)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape = (None, num_tokens))
    decoder_outputs, _, _  = tf.keras.layers.LSTM(num_tokens, return_sequences=True, return_state=True)(decoder_inputs, initial_state = encoder_states)

    print(decoder_outputs.shape)

    seq_2_seq_autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    encoder = Model(encoder_inputs, encoder_outputs)

    # match input and ouput size
    # fin_output = Dense(num_tokens, activation='softmax')(dec_output) # secondo me questa linea non è corretta, io la toglierei e forse sarebbe apposto così

    # sequence_autoencoder = Model([enc_input, dec_input], fin_output)

    # sequence_autoencoder = Model(enc_input, dec_input)

    # encoder = Model(enc_input, enc_output)

    return seq_2_seq_autoencoder, encoder


# create the model
num_features = 26 # number of features per sample (Mel)
num_units = 5 # GRU units in encoder and decoder
autoenc, enc = rnn_model(num_features, num_units)

# creating a fake trainset set with 10 samples, just repeating one sample wav file
X_train = []
for i in range (0, 100):
    (rate, sig) = wav.read("data/wav_example.wav")
    mfcc_feat = mfcc(sig, samplerate=rate,winlen=0.025,winstep=0.01,numcep=num_features,
                     nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
                     ceplifter=22,appendEnergy=True) # capire bene come funziona questa funzione...
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
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
autoenc.compile(optimizer = opt, loss = tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])
autoenc.fit(x = [X_train, X_train_shifted], y = X_train, epochs = 2, batch_size = 32)

# link utili:
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/lstm_seq2seq/
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://keras.io/api/layers/recurrent_layers/lstm/
# https://machinelearningmastery.com/lstm-autoencoders/
