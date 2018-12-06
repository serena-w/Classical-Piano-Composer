""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Activation, Input, Concatenate, Reshape
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch.midi))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        # remove this to train on full dataset
        break

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    noteRange = 128

    # notes is a list of note and chord's represented by midi value
    pitchnames = sorted(set(notes))

    network_input = numpy.zeros((len(notes)-sequence_length, noteRange, sequence_length))
    network_output = numpy.zeros((len(notes)-sequence_length, noteRange))

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        # build network input using n-hot vectors
        for j in range(0,len(sequence_in)):
            elem = sequence_in[j]
            if elem.find('.') != -1:
                for n in elem.split('.'):
                    network_input[i][int(n)][j] = 1
            else:
                network_input[i][int(elem)][j] = 1 

        # build network output using a single n-hot vector
        if sequence_out.find('.') != -1:
            for n in sequence_out.split('.'):
                network_output[i][int(n)] = 1
        else:
            network_output[i][int(sequence_out)] = 1 

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))
    lstm1 = LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )(inputs)

    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(
        512,
        return_sequences=True
    )(dropout1)

    dropout2 = Dropout(0.3)(lstm2)
    lstm3 = LSTM(
        512
    )(dropout2)

    dense1 = Dense(256)(lstm3)
    dropout3 = Dropout(0.3)(dense1)

    dense2 = Dense(128,activation='sigmoid')(dropout3)
    model = Model(inputs = inputs, outputs = dense2)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
