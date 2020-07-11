
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
# NETWORK_MODEL_TO_LOAD = 'cnn_model1'

MODEL_VERSION_TO_LOAD = 0.1


NEW_AUDIO_FILES_DIR = 'C:/Users/admin/Desktop/HDA/final_project/kws_engine/src_nicola/kws_engine/new_audio_files'
NEW_AUDIO_FILENAME = 'five_1'

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


if RUN_ON_CLUSTER:
    TRAIN_DIR = '/nfsd/hda/DATASETS/Project_1'
else:
    # TRAIN_DIR = 'C:/Users/Leonardo/Documents/Uni/HDA/Project/speech_commands_v0.02'
    # TRAIN_DIR = 'C:/Users/Leonardo/Documents/Uni/HDA/Project/debug_dataset_020620/train'
    TRAIN_DIR = 'C:/Users/admin/Desktop/HDA/final_project/dataset/_'



# ----------------------------  MAIN --------------------------

def main(argv):

    print()

    # LETTURA DEI FILENAME E CREAZIONE DELLE LABEL
    print('Loading the new audio file...')
    print()

    # default input_shape of the encoder_mlp_classifier
    input_shape = (MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES)

    if NETWORK_MODEL_TO_LOAD == 'cnn_model1':
        input_shape = (MAX_TIMESTEPS_SPECTROGRAMS, NUM_FEATURES, 1)


    real_label = NEW_AUDIO_FILENAME.split('_')[0]
    input_spectrogram = compute_spectrogram(str(NEW_AUDIO_FILES_DIR + '/' + NEW_AUDIO_FILENAME + '.wav'), input_shape, WIN_LEN, WIN_STEP, FEATURES_TYPES[FEATURES_CHOICE])

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

        labels_counter += 1


    print('Done')
    print()

    print('Loading the trained classification model...')
    classification_model = load_model(
        './training_output/models/' + NETWORK_MODEL_TO_LOAD + '_v' + str(MODEL_VERSION_TO_LOAD) + '.h5')

    print()
    classification_model.summary()
    print('Done')
    print()

    print('Making prediction on the new audio file:')

    prediction = classification_model.predict(input_spectrogram)[0]
    predicted_label = labels_dict[np.argmax(prediction)]
    print('Real label: ' + real_label + '   --->   Predicted label: ' + predicted_label)
    print()

    if real_label == predicted_label:
        print('Correct audio file classification!')
    else:
        print('Incorrect audio file classification!')






def compute_spectrogram(filename, input_size, win_len, win_step, feature_type):

    rate, signal = wav.read(str(filename))

    # siccome sappiamo che la durata massima di un file audio è un secondo (cioè max(length(signal)) = rate)
    # tengo al massimo un secondo
    signal = signal[0:rate]
    # faccio padding con zeri fino a 1 secondo se l'audio è più corto
    signal_padded = np.pad(signal, (0, rate - signal.shape[0]))

    # print('Rate: ' + str(rate))
    # print('Signal length: ' + str(len(sig)))

    # output = np.zeros(input_size)

    num_features = input_size[1]

    if feature_type == 'cepstral':

        mfcc_features = mfcc(signal_padded, samplerate=rate, winlen=win_len, winstep=win_step, numcep=num_features,
                         nfilt=num_features, nfft=512, #calculate_nfft(rate, win_len),
                         lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)

        output = (mfcc_features / (np.amax(np.abs(mfcc_features)))).astype(dtype='float32')



    if feature_type == 'mel-spectrogram':

        mel_spectrogram, _ = fbank(signal_padded, samplerate=rate, winlen=win_len, winstep=win_step,
                         nfilt=num_features, nfft=512, #calculate_nfft(rate, win_len),
                         lowfreq=0, highfreq=None, preemph=0.97)

        output = ((2 * mel_spectrogram / np.amax(mel_spectrogram)) - 1).astype(dtype='float32')



    if feature_type == 'log-mel-spectrogram':

        mel_spectrogram = logfbank(signal_padded, samplerate=rate, winlen=win_len, winstep=win_step,
                                      nfilt=num_features, nfft=512,  # calculate_nfft(rate, win_len),
                                      lowfreq=0, highfreq=None, preemph=0.97)

        output = (mel_spectrogram / (np.amax(np.abs(mel_spectrogram)))).astype(dtype='float32')



    if feature_type == 'mel-spectrogram-Audeep':

        _, _, p_s = power_spectrum(signal, rate, int(rate * win_len), int(rate * win_step))
        mel_spectrogram = mel_spectrum(p_s, None, rate, int(rate * win_len), num_features).astype(dtype='float32')
        print(mel_spectrogram.shape)

        output = ((2 * np.rot90(mel_spectrogram) / np.amax(mel_spectrogram)) - 1).astype(dtype='float32')


    # this means that we are using a cnn model, because it expects a 3D input
    if len(input_size == 3):
        output = output.reshape(input_size)


    return output




if __name__ == '__main__':
    app.run(main)
