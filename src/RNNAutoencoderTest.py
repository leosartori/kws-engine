
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
    enc_input = Input(shape = (None, num_tokens))
    enc_output, enc_state = tf.keras.layers.GRU(num_units, return_state=True) \
        (enc_input) # input = [batch, timesteps, feature]

    # decoder
    dec_input = Input(shape = (None, num_tokens))
    dec_output = tf.keras.layers.GRU(num_units, return_sequences=True)(dec_input, initial_state = enc_state)

    # match input and ouput size
    fin_output = Dense(num_tokens, activation='softmax')(dec_output)

    sequence_autoencoder = Model([enc_input, dec_input], fin_output)

    encoder = Model(enc_input, enc_output)

    return sequence_autoencoder, encoder



# create the model
num_features = 26 # number of features per sample (Mel)
num_units = 3 # GRU units in encoder and decoder
autoenc, enc = rnn_model(num_features, num_units)





# creating a fake trainset set with 10 samples, just repeating one sample wav file
X_train = []
for i in range (0, 10):
    (rate, sig) = wav.read("wav_example.wav")
    mfcc_feat = mfcc(sig, samplerate=rate,winlen=0.025,winstep=0.01,numcep=num_features,
                     nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
                     ceplifter=22,appendEnergy=True)
    X_train.append(mfcc_feat)
X_train = np.array(X_train)

# decoder input, as shifted encoder input
X_train_shifted = np.zeros(X_train.shape)
# loop in timesteps
for sample in range(0, X_train.shape[0]):
    for timestep in range(0, X_train.shape[1] - 1):
        X_train_shifted[sample, timestep + 1, :] = X_train[sample, timestep, :]

# train the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
autoenc.compile(optimizer = opt, loss = tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])
autoenc.fit(x = [X_train, X_train_shifted], y = X_train, epochs = 100, batch_size = 256)

